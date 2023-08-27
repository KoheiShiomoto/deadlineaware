# 見習いデータサイエンティストの隠れ家
# 2022-03-13
# Pytorchを使った深層強化学習！〜Actor Criticの構築〜
# https://www.dskomei.com/entry/2022/03/13/114756


# Actor Critic は、状態から行動を予測するモデルと、状態から価値（累積報酬）を予測する２つのモデルから構成されます。Actor モデルが学習する際に、Critic が予測した状態価値の影響を受けます。Actor だけでは、価値に関係なく行われた行動を予測するばかりなので、強化学習の意義である価値最大化の方策モデルができません。Critic に状態を評価してもらい、その評価に基づいて Actor の予測確率を変えることが大事です。


# 必要なモジュールをインポートします。
from pathlib import Path
import math
import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns
import settings

# 実験の際にできたモデルや結果ファイルを保存するためのディレクトリを作成しておきます。
model_dir_path = Path('mdoel')
result_dir_path = Path('result')
if not model_dir_path.exists():
    model_dir_path.mkdir()
if not result_dir_path.exists():
    result_dir_path.mkdir()

# 強化学習を行うためのゲーム環境を設定します。今回行うのは「CartPole」というゲームです。カートの上に棒が立てられており、この棒が倒れないようにカートを動かし、何ステップ保てるかを競うゲームです。このとき、ゲームを行う人が取りうる行動はカートを動かす向きの選択であり、「左 or 右」のどちらを動かすかです。なので、深層強化学習のモデルとしては、状態の情報を受け取り、”左”と”右”の選択確率を出力するモデルを作ります。

# 強化学習を簡単に行うために、OpenAI Gymを使います。OpenAI Gymに関してはこちらの記事が参考になります。
# OpenAI Gym 入門
# https://qiita.com/ishizakiiii/items/75bc2176a1e0b65bdd16

# Pythonでは以下のようにしてゲーム環境を立ち上げます。    

env = gym.make('CartPole-v1')
env.seed(settings.seed)

# モデルの構築

# 作成するモデルは、状態の情報を入力として、行動確率と状態価値を出力します。重みが共有されており、出力層で行動確率と状態価値のレイヤーを分けています。この一つのモデルで Actor と Critic を表しています。

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.affine = nn.Linear(4, 128)

        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.saved_rewards = []

    def forward(self, x):
        x = F.relu(self.affine(x))

        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_prob, state_values

# Actor Critic モデルを使って行動を選択する関数を作ります。このとき、モデルが予測した行動確率を使ってランダムに行動選択するようにしています。これは、モデルが予測した最大確率の行動を選択していると、モデルの学習時のデータが偏ってしまい、それを防ぐためです。そして、選択した行動の対数確率を「saved_actions」に保存しています。

def select_action(model, state, device):
    state = torch.from_numpy(state).float()

    probs, state_value = model(state.to(device))
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append((m.log_prob(action), state_value))

    return action.item(), state_value

# Actor Critic の学習関数

# Actor Critic を学習する関数を作ります。最初に各ステップの累積報酬を求めます。そして、各ステップごとに、累積報酬と Critic の予測である価値との差分を求め、Actor の損失値と Critic の損失値を計算しています。この2つの損失値を足し合わせたものを最小化します。最後に、このエピソードで獲得した報酬と選択した行動の対数確率のリストの中身を削除しています。

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

# Actor Critic の学習の実行

# ここまでで準備は全て終わったので、あとはそれぞれを実行していくだけです。10エピソードごとの獲得報酬の平均が500に達したら学習を終えるようにしています。つまり、10エピソード連続で500ステップ棒を倒さなければ、学習が終了します。これは厳しい設定だと思うので、強化学習の設定に合わせて考える必要があります。

torch.manual_seed(settings.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Policy()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()

reward_result = []
reward_displays = []
loss_result = []
loss_display = []
for episode in range(1, settings.epochs + 1):

    state = env.reset()
    episode_reward = 0
    for t in range(1, 10000):

        action, _ = select_action(model=model, state=state, device=device)
        state, reward, done, _ = env.step(action)

        if settings.flag_env_render:
            env.render()

        model.saved_rewards.append(reward)
        episode_reward += reward
        if done:
            break

    reward_displays.append(episode_reward)

    loss = learn_model(model=model, gamma=settings.gamma, optimizer=optimizer, device=device)
    loss = loss.detach().item()
    loss_display.append(loss)

    reward_result.append(episode_reward)
    loss_result.append(loss)

    if env.spec.max_episode_steps == np.mean(reward_displays):
        print('Game Win !!')
        break

    if episode % settings.display_loop == 0:
        print('Episode {}\tMean Loss: {:.2f}\tMean Reward: {:.0f}'.format(episode, np.mean(loss_display), np.mean(reward_displays)))
        reward_displays = []
        loss_display = []


model = model.to('cpu')
torch.save(model.state_dict(), model_dir_path.joinpath('actor_critic.pth'))

result = pd.DataFrame({
    'episode': np.arange(1, len(reward_result) + 1),
    'reward': reward_result,
    'loss': loss_result
})
result.to_csv(result_dir_path.joinpath('actor_critic_cartpole_result.csv'), index=False)

# 学習済みモデルの検証

# モデルの学習が完了したので、学習済みモデルの結果を確認していきます。


# 学習時の獲得報酬の推移

# まずは、学習過程を確認するために、エピソードごとの獲得報酬の推移を見てみます。

result_data = pd.read_csv(result_dir_path.joinpath('actor_critic_cartpole_result.csv'))

fig = plt.figure(figsize=(12, 6), facecolor='white')

ax = fig.add_subplot(1, 2, 1)
g = sns.lineplot(
    data=result_data,
    x='episode', y='reward',
    ax=ax
)
plt.title('報酬の推移', fontsize=18, weight='bold')
plt.ylabel('')

ax = fig.add_subplot(1, 2, 2)
g = sns.lineplot(
    data=result_data.assign(group=lambda x: x.episode.map(lambda y: math.floor((y - 1) / 10))),
    x='group', y='reward',
    ax=ax
)
plt.title('10エピソードごとの報酬平均値の推移', fontsize=18, weight='bold')
plt.ylabel('')

plt.tight_layout()
plt.savefig(result_dir_path.joinpath('actor_critic_cartpolcar_learning_reward.png'), dpi=300)

# 学習済みモデルを使ったゲームプレイ

# 学習済みモデルを使って、「CartPole」をプレイします。まずは、行動選択関数を作ります。ここでは、Actor の最大確率の行動を選択するようにしています。

def select_action_valid(model, state, device):

    with torch.no_grad():
        state = torch.from_numpy(state).float()
        probs, state_value = model(state.to(device))

    return probs.argmax().item(), state_value

# 学習済みモデルをロードします。

model = Policy()
model.load_state_dict(torch.load(model_dir_path.joinpath('actor_critic.pth')))
model.eval()

# 学習済みモデルを使って、ゲームをプレイします。

state = env.reset()
episode_reward = 0
done = False
while not done:

    action, state_value = select_action_valid(model=model, state=state, device=device)
    state, reward, done, _ = env.step(action)

    env.render()
    episode_reward += reward

print('Game End Reward: {:.0f}'.format(episode_reward))

