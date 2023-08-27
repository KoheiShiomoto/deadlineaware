import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from env_deadline_aware import EnvDeadlineAware
from env_deadline_aware import convert_obs_to_state
from env_deadline_aware import count_vacant

from scheduler import select_job_FCFS
from scheduler import select_job_EDF
from tsdb import TsDatabase
#
# 2023-07-19
#
from tsdb import EvalBusyPeriod


class Agent():
    def __init__(self,
                 env,
                 seed,
                 #
                 pjName,
                 tmax):
        #
        self.env = env
        self.seed = seed
        #
        self.pjName = pjName
        self.tmax = tmax
        # 実験の際にできた結果ファイルを保存するためのディレクトリを作成
        self.output_dir_path = Path('output')
        if not self.output_dir_path.exists():
            self.output_dir_path.mkdir()
        self.ofileName_base = "output/odata_"+pjName
        return
                 
    def test(self):
        ntick = 10
        tick = int(self.tmax/ntick)
        observation,_ = self.env.reset()
        nact = self.env.get_nact()
        state, num_state_elem = convert_obs_to_state(observation)
        logger = TsDatabase(ofileName_base = self.ofileName_base)
        self.model_dir_path = Path('model')
        self.result_dir_path = Path('result')
        self.eval_busyperiod = EvalBusyPeriod(ofileName_base = self.ofileName_base)
        episode_reward = 0
        for t in range(self.tmax):
            idx = self.select_job(observation)
            observation, reward, terminated, truncated, info = self.env.step(action=idx)
            self.eval_busyperiod.step(reward, info)
            episode_reward += reward
            logger.add([("AccumReward",episode_reward)])
            time, num_jobs, qlen = self.env.get_status()
            logger.add([("time",time),("num_jobs",num_jobs),("qlen",qlen)])
            if terminated:
                break
        #
        logger.plot()
        logger.to_csv()
        self.eval_busyperiod.to_csv()
        self.eval_busyperiod.plot()

class Agent_EDF(Agent):
    def __init__(self,
                 env,
                 seed,
                 #
                 pjName,
                 tmax):
        #
        super().__init__(env = env,
                         seed = seed,
                         pjName = pjName,
                         tmax = tmax)

    def select_job(self,
                   observation):
        selected_index = -1
        edf_time = float('inf')
        for index, job in enumerate(observation):
            # # full attributes
            # observation.append([arriving_time, deadline_length, deadline, job_size, job_size_remain]) # full attributes
            arriving_time = job.pop(0)
            deadline_length = job.pop(0)
            deadline = job.pop(0)
            job_size = job.pop(0)
            job_size_remain = job.pop(0)
            if job_size_remain > 0 and deadline <= edf_time: # job_sizeが0より大きいものが有効なジョブ
                edf_time = deadline
                selected_index = index
        return selected_index
        
class Agent_FCFS(Agent):
    def __init__(self,
                 env,
                 seed,
                 #
                 pjName,
                 tmax):
        #
        super().__init__(env = env,
                         seed = seed,
                         pjName = pjName,
                         tmax = tmax)

    def select_job(self,
                   observation):
        if len(observation) == 0:
            return -1
        # #
        # # full attributes
        # observation.append([arriving_time, deadline_length, deadline, job_size, job_size_remain]) # full attributes
        min_arriving_time = float('inf')
        selected_index = -1
        for index, job in enumerate(observation):
            arriving_time = job.pop(0)
            deadline_length = job.pop(0)
            deadline = job.pop(0)
            job_size = job.pop(0)
            job_size_remain = job.pop(0)
            if job_size_remain > 0 and arriving_time <= min_arriving_time: # job_sizeが0より大きいものが有効なジョブ
                min_arriving_time = arriving_time
                selected_index = index
        return selected_index


class Policy(nn.Module):
    def __init__(self, nact, num_state_elem):
        super(Policy, self).__init__()
        nn_input_size = nact*num_state_elem
        #
        self.affine = nn.Linear(nn_input_size, 256)
        self.action_head = nn.Linear(256, nact)
        self.value_head = nn.Linear(256, 1)
        #
        # self.affine = nn.Linear(nn_input_size, 128)
        # self.action_head = nn.Linear(128, nact)
        # self.value_head = nn.Linear(128, 1)
        #
        # self.affine1 = nn.Linear(nn_input_size, 256)
        # self.affine2 = nn.affine1(256, 256)
        # self.action_head = nn.affine2(256, nact)
        # self.value_head = nn.affine2(256, 1)

        self.saved_actions = []
        self.saved_rewards = []

    def forward(self, x):
        x = F.relu(self.affine(x))

        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_prob, state_values


class Agent_RL(Agent):
    def __init__(self,
                 env,
                 tmax,
                 seed,
                 pjName,
                 #
                 modelName,
                 gamma = 0.99,
                 lr = 1.0e-5):
        super().__init__(env = env,
                         seed = seed,
                         pjName = pjName,
                         tmax = tmax)
        # 以下、強化学習Agentに必要な設定
        self.modelName = modelName
        self.gamma = gamma
        self.lr = lr
        # 実験の際にできたモデルや結果ファイルを保存するためのディレクトリを作成
        self.model_dir_path = Path('model')
        self.result_dir_path = Path('result')
        if not self.model_dir_path.exists():
            self.model_dir_path.mkdir()
        if not self.result_dir_path.exists():
            self.result_dir_path.mkdir()
        return
                 
    def learn_model(self,
                    model,
                    gamma,
                    optimizer,
                    device):
        R = 0
        returns = []
        for r in model.saved_rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-06)
        policy_losses = []
        value_losses = []
        for (log_prob, value), R in zip(model.saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del model.saved_rewards[:]
        del model.saved_actions[:]
        return loss

    def test(self):
        ntick = 10
        tick = int(self.tmax/ntick)
        #
        observation,_ = self.env.reset()
        nact = self.env.get_nact()
        state, num_state_elem = convert_obs_to_state(observation)
        #
        logger = TsDatabase(ofileName_base = self.ofileName_base)
        self.model_dir_path = Path('model')
        self.result_dir_path = Path('result')
        self.eval_busyperiod = EvalBusyPeriod(ofileName_base = self.ofileName_base)
        # 学習済みモデルをロード
        model = Policy(nact=nact, num_state_elem=num_state_elem)
        model.load_state_dict(torch.load(self.model_dir_path.joinpath(f'ac_model_{self.modelName}.pth')))
        model.eval()
        #
        torch.manual_seed(self.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        #
        episode_reward = 0
        for t in range(self.tmax):
            state, _ = convert_obs_to_state(observation)
            idx, _ = self.select_action_valid(model=model, state=state, device=device)
            observation, reward, terminated, truncated, info = self.env.step(action=idx)
            self.eval_busyperiod.step(reward, info)

            episode_reward += reward
            logger.add([("AccumReward",episode_reward)])
            time, num_jobs, qlen = self.env.get_status()
            logger.add([("time",time),("num_jobs",num_jobs),("qlen",qlen)])
            if terminated:
                break
        #
        logger.plot()
        logger.to_csv()
        #
        self.eval_busyperiod.to_csv()
        self.eval_busyperiod.plot()

    def train(self,
              num_episodes):
        ntick = 10
        tick = int(self.tmax/ntick)
        torch.manual_seed(self.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        observation,_ = self.env.reset()
        nact = self.env.get_nact()
        state, num_state_elem = convert_obs_to_state(observation)
        model = Policy(nact=nact, num_state_elem=num_state_elem)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        eps = np.finfo(np.float32).eps.item()
        reward_result = []
        loss_result = []
        # for episode in tqdm(range(num_episodes), leave=False):
        for episode in tqdm(range(num_episodes)):
            observation,_ = self.env.reset()
            episode_reward = 0
            terminated = False
            while terminated == False :
            # for t in range(self.tmax):
                state, _ = convert_obs_to_state(observation)
                idx, _ = self.select_action(model=model, state=state, device=device)
                observation, reward, terminated, truncated, info = self.env.step(action=idx)
                model.saved_rewards.append(reward)
                episode_reward += reward
                if terminated:
                    break
            loss = self.learn_model(model=model, gamma=self.gamma, optimizer=optimizer, device=device)
            loss = loss.detach().item()
            reward_result.append(episode_reward)
            loss_result.append(loss)
        #
        model = model.to('cpu')
        torch.save(model.state_dict(), self.model_dir_path.joinpath(f'ac_model_{self.pjName}.pth'))
        result = pd.DataFrame({
            'episode': np.arange(1, len(reward_result) + 1),
            'reward': reward_result,
            'loss': loss_result
        })
        result.to_csv(self.result_dir_path.joinpath(f'ac_result_{self.pjName}_trained_reward.csv'), index=False)
        self.plot_training_raw()
        self.plot_training_ave10ep()
        self.plot_training_2in1()

    def plot_training_raw(self):
        # plt.clf()
        result_data = pd.read_csv(self.result_dir_path.joinpath(f'ac_result_{self.pjName}_trained_reward.csv'))
        g = sns.lineplot(
            data=result_data,
            x='episode', y='reward'
        )
        g.axes.set_ylim(0,)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Reward", fontsize=14)
        plt.tick_params(labelsize=12)
        plt.savefig(self.result_dir_path.joinpath(f'ac_gh_{self.pjName}_trained_reward_raw.pdf'))
        plt.show()

    def plot_training_ave10ep(self):
        # plt.clf()
        result_data = pd.read_csv(self.result_dir_path.joinpath(f'ac_result_{self.pjName}_trained_reward.csv'))
        g = sns.lineplot(
            data=result_data.assign(group=lambda x: x.episode.map(lambda y: math.floor((y - 1) / 10))),
            x='group', y='reward'
        )
        g.axes.set_ylim(0,)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Reward", fontsize=14)
        plt.tick_params(labelsize=12)
        plt.savefig(self.result_dir_path.joinpath(f'ac_gh_{self.pjName}_trained_reward_ave10ep.pdf'))
        plt.show()

    def plot_training_2in1(self):
        result_data = pd.read_csv(self.result_dir_path.joinpath(f'ac_result_{self.pjName}_trained_reward.csv'))
        fig = plt.figure(figsize=(12, 6), facecolor='white')
        ax = fig.add_subplot(1, 2, 1)
        g = sns.lineplot(
            data=result_data,
            x='episode', y='reward',
            ax=ax
        )
        g.axes.set_ylim(0,)
        plt.title('Reward as a function of time', fontsize=16, weight='bold')
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Reward", fontsize=14)
        plt.tick_params(labelsize=12)
        ax = fig.add_subplot(1, 2, 2)
        g = sns.lineplot(
            data=result_data.assign(group=lambda x: x.episode.map(lambda y: math.floor((y - 1) / 10))),
            x='group', y='reward',
            ax=ax
        )
        g.axes.set_ylim(0,)
        plt.title('Reward as a function of time\n (average per 10 episodes)', fontsize=16, weight='bold')
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Reward", fontsize=14)
        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(self.result_dir_path.joinpath(f'ac_gh_{self.pjName}_trained_reward.pdf'))
        plt.show()

        
class Agent_AC(Agent_RL):
    def __init__(self,
                 env,
                 tmax,
                 seed,
                 pjName,
                 #
                 modelName,
                 gamma = 0.99,
                 lr = 1.0e-5):
        #
        super().__init__(env = env,
                         tmax = tmax,
                         seed = seed,
                         pjName = pjName,
                         #
                         modelName = modelName,
                         gamma = gamma,
                         lr = lr)

    # 学習のための行動選択関数
    def select_action(self,
                      model,
                      state,
                      device):
        state = np.array(state, dtype=float).flatten()
        state = torch.from_numpy(state).float()
        probs, state_value = model(state.to(device))
        m = Categorical(probs)
        action = m.sample()
        model.saved_actions.append((m.log_prob(action), state_value))
        return action.item(), state_value

    # テストのための行動選択関数
    # Actor の最大確率の行動を選択
    def select_action_valid(self,
                            model,
                            state,
                            device):
        vacant, nact = count_vacant(state)
        state = np.array(state, dtype=float).flatten()
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            probs, state_value = model(state.to(device))
        probs[-vacant:] = 0.0
        if vacant == nact:
            idx = -1
        else:
            idx = probs.argmax().item()
        return idx, state_value
