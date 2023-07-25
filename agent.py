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


class Policy(nn.Module):
    def __init__(self, nact, num_state_elem):
        super(Policy, self).__init__()
        nn_input_size = nact*num_state_elem
        self.affine = nn.Linear(nn_input_size, 256)
        self.action_head = nn.Linear(256, nact)
        self.value_head = nn.Linear(256, 1)

        self.saved_actions = []
        self.saved_rewards = []

    def forward(self, x):
        x = F.relu(self.affine(x))

        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_prob, state_values

# 学習のための行動選択関数
def select_action_RLAC(model, state, device):
    state = np.array(state, dtype=float).flatten()
    state = torch.from_numpy(state).float()
    probs, state_value = model(state.to(device))
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append((m.log_prob(action), state_value))

    return action.item(), state_value

# テストのための行動選択関数
# Actor の最大確率の行動を選択
def select_action_valid_RLAC(model, state, device):
    vacant = count_vacant(state)
    state = np.array(state, dtype=float).flatten()
    with torch.no_grad():
        state = torch.from_numpy(state).float()
        probs, state_value = model(state.to(device))
    probs[-vacant:] = 0.0
    return probs.argmax().item(), state_value


def learn_model(model, gamma, optimizer, device):

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

class Agent_AC():
    def __init__(self,
                 env,
                 seed,
                 algorithm,
                 #
                 pjName,
                 tmax,
                 gamma = 0.99,
                 lr = 1.0e-5):
        #
        self.env = env
        self.seed = seed
        self.algorithm = algorithm
        self.gamma = gamma
        self.lr = lr
        #
        self.pjName = pjName
        self.tmax = tmax
        
        self.ofileName_base = "output/odata_"+pjName

        self.model_dir_path = Path('model')
        self.result_dir_path = Path('result')
        self.output_dir_path = Path('output')
        if not self.model_dir_path.exists():
            self.model_dir_path.mkdir()
        if not self.result_dir_path.exists():
            self.result_dir_path.mkdir()
        if not self.output_dir_path.exists():
            self.output_dir_path.mkdir()

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

        if self.algorithm == "AC":
            model = Policy(nact=nact, num_state_elem=num_state_elem)
            model.load_state_dict(torch.load(self.model_dir_path.joinpath(f'ac_model_{self.pjName}.pth')))
            model.eval()
            torch.manual_seed(self.seed)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
        #
        episode_reward = 0
        for t in range(self.tmax):
            if self.algorithm == "FCFS":
                idx = select_job_FCFS(observation)
            elif self.algorithm == "EDF":
                idx = select_job_EDF(observation)
            elif self.algorithm == "AC":
                state, _ = convert_obs_to_state(observation)
                idx, _ = select_action_valid_RLAC(model=model, state=state, device=device)
            else:
                print("Algorithm Not defined.")
                sys.exit(1)
            self.env.show_status()
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


    def train(self,
              num_episodes):
        ntick = 10
        tick = int(self.tmax/ntick)

        # Actor Critic の学習の実行
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

        for episode in tqdm(range(num_episodes)):
            observation,_ = self.env.reset()
            episode_reward = 0
            terminated = False
            while terminated == False :
            # for t in range(self.tmax):
                if self.algorithm == "AC":
                    state, _ = convert_obs_to_state(observation)
                    idx, _ = select_action_RLAC(model=model, state=state, device=device)
                else:
                    print("Algorithm Not defined.")
                    sys.exit(1)
                observation, reward, terminated, truncated, info = self.env.step(action=idx)
                model.saved_rewards.append(reward)
                episode_reward += reward
                if terminated:
                    break
            loss = learn_model(model=model, gamma=self.gamma, optimizer=optimizer, device=device)
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
        print(result)
        result.to_csv(self.result_dir_path.joinpath(f'ac_result_{self.pjName}_trained_reward.csv'), index=False)
        #
        self.plot_training_raw()
        self.plot_training_ave10ep()
        self.plot_training_2in1()

    def plot_training_raw(self):
        plt.clf()
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
        plt.clf()
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


