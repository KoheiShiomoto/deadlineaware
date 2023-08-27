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
    # #
    # # 2023-07-12
    # vacant = count_vacant(state)
    # # 2023-07-12
    # #
    state = np.array(state, dtype=float).flatten()
    with torch.no_grad():
        state = torch.from_numpy(state).float()
        probs, state_value = model(state.to(device))
        #
        #
        print(probs)
        #
        #
    # #
    # # 2023-07-12
    # probs[-vacant:] = 0.0
    # # 2023-07-12
    # #
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

        # 実験の際にできたモデルや結果ファイルを保存するためのディレクトリを作成
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
        # print(f"num_state_elem:{num_state_elem}, state:{state}")

        model = Policy(nact=nact, num_state_elem=num_state_elem)
        model = model.to(device)
        # optimizer = optim.Adam(model.parameters(), lr=3e-2)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        eps = np.finfo(np.float32).eps.item()

        reward_result = []
        loss_result = []

        # for episode in range(num_episodes):
        # for episode in tqdm(range(num_episodes), leave=False):
        for episode in tqdm(range(num_episodes)):
            observation,_ = self.env.reset()
            episode_reward = 0
            for t in range(self.tmax):
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
        result.to_csv(self.result_dir_path.joinpath(f'ac_result_trained_reward_{self.pjName}.csv'), index=False)
        #
        # 学習時の獲得報酬の推移
        # まずは、学習過程を確認するために、エピソードごとの獲得報酬の推移を確認する
        result_data = pd.read_csv(self.result_dir_path.joinpath(f'ac_result_trained_reward_{self.pjName}.csv'))
        fig = plt.figure(figsize=(12, 6), facecolor='white')
        ax = fig.add_subplot(1, 2, 1)
        g = sns.lineplot(
            data=result_data,
            x='episode', y='reward',
            ax=ax
        )
        plt.title('Reward as a function of time', fontsize=18, weight='bold')
        plt.ylabel('')
        #
        ax = fig.add_subplot(1, 2, 2)
        g = sns.lineplot(
            data=result_data.assign(group=lambda x: x.episode.map(lambda y: math.floor((y - 1) / 10))),
            x='group', y='reward',
            ax=ax
        )
        plt.title('Reward as a function of time\n (average per 10 episodes)', fontsize=18, weight='bold')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.savefig(self.result_dir_path.joinpath(f'ac_gh_trained_reward_{self.pjName}.pdf'))
        plt.show()
        #

    def test(self):

        #
        ntick = 10
        tick = int(self.tmax/ntick)
        #

        observation,_ = self.env.reset()
        nact = self.env.get_nact()
        state, num_state_elem = convert_obs_to_state(observation)
        # print(f"num_state_elem:{num_state_elem}, state:{state}")

        logger = TsDatabase(ofileName_base = self.ofileName_base)
        #
        self.model_dir_path = Path('model')
        self.result_dir_path = Path('result')
        if self.algorithm == "AC":
            # 学習済みモデルをロード
            model = Policy(nact=nact, num_state_elem=num_state_elem)
            # model = Policy()
            model.load_state_dict(torch.load(self.model_dir_path.joinpath(f'ac_model_{self.pjName}.pth')))
            model.eval()
            #
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
            #
            #
            print(f"#################################idx:{idx}")
            self.env.show_status()
            #
            #
            observation, reward, terminated, truncated, info = self.env.step(action=idx)
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

