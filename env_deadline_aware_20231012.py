# import sys
# from typing import (
#     TYPE_CHECKING,
#     Any,
#     Dict,
#     Generic,
#     List,
#     Optional,
#     SupportsFloat,
#     Tuple,
#     TypeVar,
#     Union,
# )

# import numpy as np

# ObsType = TypeVar("ObsType")
# ActType = TypeVar("ActType")
# RenderFrame = TypeVar("RenderFrame")


import math
import copy
import pickle
from pathlib import Path

from jobset import get_list_job_info, Job, SelfJobSet, TraceJobSet, DrillJobSet
from tsdb import EvalBusyPeriod


def show_job_queue(job_queue):
    for item in job_queue:
        arrival_time, deadline_length, deadline, job_size, job_size_remain = item.get_job_info()
        print(f"# arrival_time:{arrival_time}, deadline_length:{deadline_length}, deadline:{deadline}, job_size:{job_size}, job_size_remain:{job_size_remain}")
    return 


def clip(value):
    lowerbound = -100
    upperbound = 100
    if value < lowerbound:
        value = lowerbound
    elif value > upperbound:
        value = upperbound
    return value

def count_vacant(state):
    cnt = 0
    size = len(state)
    for item in state:
        deadline_remain, job_size_remain = item[0], item[1]
        if job_size_remain <= 0:
            cnt += 1
    # if cnt == size:
    #     cnt -= 1
    return cnt, size

def convert_obs_to_state(observation):
    state = []
    # 
    # l = [0,0,0,0,0] # [arrival_time, deadline_length, deadline_remain, job_size, job_size_remain] # full attributes
    l = [0,0] # [deadline_remain, job_size_remain] # subset1 attributes
    # l = [0,0,0] # [arrival_time, deadline_remain, job_size_remain] # subset2 attributes
    # l = [0,0,0,0] #[arrival_time, deadline_remain, job_size, job_size_remain] # subset3 attributes
    #
    for item in observation:
        arrival_time, deadline_length, deadline_remain, job_size, job_size_remain = item[0], item[1], item[2], item[3], item[4] 
        #
        # dummy ジョブの場合はstateは強制的に[0,0]にする -> あまり効果はなさそう。。。
        if job_size == 0:
            deadline_remain = 0
            job_size_remain = 0
        #
        # l = [arrival_time, deadline_length, deadline_remain, job_size, job_size_remain] # full attributes
        deadline_remain = clip(deadline_remain)
        l = [deadline_remain, job_size_remain] # subset1 attributes most useful pattern? 2
        # l = [arrival_time, deadline_remain, job_size_remain] # subset2 attributes for FCFS testing 3
        # l = [arrival_time, deadline_remain, job_size, job_size_remain] # subset3 attributes 4
            
        num_state_elem = len(l)
        state.append(l)
    return state, num_state_elem

def make_observation(list_jobs, time):
    observation = []
    # #
    l = [0,0,0,0,0] # [arrival_time, deadline_length, deadline_remain, job_size, job_size_remain] # full attributes
    # l = [0,0] # [deadline_remain, job_size_remain] # subset1 attributes
    # l = [0,0,0] # [arrival_time, deadline_remain, job_size_remain] # subset2 attributes
    # l = [0,0,0,0] #[arrival_time, deadline_remain, job_size, job_size_remain] # subset3 attributes
    #
    num_obs_elem = 0
    num_jobs = 0
    sum_job_size_remain = 0
    for item in list_jobs:
        arrival_time, deadline_length, deadline, job_size, job_size_remain = item.get_job_info()
        deadline_remain = deadline-time
        #
        l = [arrival_time, deadline_length, deadline_remain, job_size, job_size_remain] # full attributes
        # l = [deadline_remain, job_size_remain] # subset1 attributes most useful pattern? 2
        # l = [arrival_time, deadline_remain, job_size_remain] # subset2 attributes for FCFS testing 3
        # l = [arrival_time, deadline_remain, job_size, job_size_remain] # subset3 attributes 4
        #
        num_obs_elem = len(l)
        num_jobs += 1
        sum_job_size_remain += job_size_remain
        observation.append(l)
    return observation, num_obs_elem, num_jobs, sum_job_size_remain


class EnvDeadlineAware():
    def __init__(self,
                 bplist_name,
                 jobset_name,
                 isStream = True,
                 isTraceJobSet = False,
                 isDrillJobSet = False,
                 f_name = "data_jobset.txt",
                 tmax = 300,
                 seed = 0,
                 nact = 8,
                 mu = 1.0,
                 lambda_ = 0.08,
                 beta = 10.0,
                 alpha = 2.0,
                 perf_thresh = 1.0,
                 num_bp = 1000000,
                 nrep = 10000): 
        """
        Args:
        num_bpは1000000以上の値にしておくと全てを選択する仕様
        nrepはDrillパターンをデフォルトで10000回行う
        Returns:
        """
        self.output_dir_path = Path('output')
        if not self.output_dir_path.exists():
            self.output_dir_path.mkdir()
        self.ofileName_base = "output/odata_"+bplist_name

        self.pkl_dir_path = Path('pkl')
        if not self.pkl_dir_path.exists():
            self.pkl_dir_path.mkdir()
        pkl_fname = self.pkl_dir_path.joinpath(f'jobset_{jobset_name}.pkl')
        #
        self.isStream = isStream
        self.f_name = f_name
        self.tmax = tmax # 時計の最大値
        self.time = 0 # 時計の初期化
        self.seed = seed # 乱数の種
        self.nact = nact # ニューラルネットに状態として与えることができるジョブ数の最大値
        self.mu = mu # 報酬計算のための減衰係数 exp(-mu*delay)
        self.lambda_ = lambda_ # ポアソン過程のパラメータ（1スロットに到着するジョブ数を決める）
        self.beta = beta # 平均ジョブサイズ
        self.alpha = alpha # 平均デッドライン長
        self.perf_thresh = perf_thresh
        self.num_bp = num_bp
        self.nrep = nrep
        #
        self.job_queue = [] # 現在、処理を待っているジョブの待ち行列
        # 最初にここでJobSetを作っておく
        if isTraceJobSet == True:
            self.jobset = TraceJobSet(tmax = self.tmax,
                                      f_name = self.f_name)
        elif isDrillJobSet == True:
            self.eval_busyperiod = EvalBusyPeriod(ofileName_base = self.ofileName_base)
            with open(pkl_fname, 'rb') as p:
                jobset = pickle.load(p)
            print(f"perf_thresh:{self.perf_thresh}")
            #
            # 2023-0907
            # list_bp = self.eval_busyperiod.from_csv(perf_thresh=self.perf_thresh)
            list_bp = self.eval_busyperiod.from_csv(perf_thresh=self.perf_thresh,
                                                    sample_size = num_bp)
            # 2023-0907
            #
            # jobset.show_config()
            # print(f"list_bp:{list_bp}")
            self.jobset = DrillJobSet(jobset = jobset,
                                      list_bp = list_bp,
                                      nrep = nrep)
            self.jobset.show_drill_config()
        else:
            self.jobset = SelfJobSet(tmax = self.tmax,
                                 seed = self.seed,
                                 isStream = self.isStream,
                                 lambda_ = self.lambda_,
                                 alpha = self.alpha,
                                 beta = self.beta)
            with open(pkl_fname, 'wb') as p:
                pickle.dump(self.jobset, p)
        #
        self.time = 0
        self.tmax = self.jobset.get_tmax()
        self.qlen = 0
        #
        if (len(self.job_queue) >= self.nact): # 待ち行列にいるジョブ数がnact以上の場合は先頭からnactを取ってくる
            nact_job_queue = copy.copy(self.job_queue[0:self.nact])
        else: # 待ち行列にいるジョブ数がnactより小さい場合はその分をダミーデータで埋める
            nact_job_queue = copy.copy(self.job_queue)
            dummy_job = Job()
            dummy_job.turn_dummy()
            for i in range(self.nact-len(self.job_queue)):
                nact_job_queue.append(dummy_job)
        observation, self.num_obs_elem, _, _ = make_observation(list_jobs = nact_job_queue, time = self.time)
        
    # def step(self,action):
    # # 1ステップ進めるときの更新処理．actionは選択されたジョブを指す．ジョブの到着処理もここで行う？
    #     return state, reward, done, 
    
    # OpenAI GymのAPI
    # https://github.com/openai/gym/blob/master/gym/core.py#L36
    # def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(observation, reward, terminated, truncated, info)`.

        Args:
            action (ActType): an action provided by the agent

        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.

            (deprecated)
            done (bool): A boolean value for if the episode has ended, in which case further :meth:`step` calls will return undefined results.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        #
        index = int(action)
        reward = 0
        completed_job_size = 0
        completed_job_num = 0
        done_idx = -1 # 待ち行列から転送が完了したジョブ番号を覚えておく
        # if index >= len(self.job_queue): # 行列長以上のインデックスを指した場合は-1にする
        #     index = -1
        # #
        for i in range(len(self.job_queue)): # 待ち行列にいるジョブについて処理をする
            if i == index: # indexで示された1個のジョブを選択してデータ転送する
                delay, deadline_length, job_size, done = self.job_queue[i].step(served=True,time=self.time)
                #
                self.qlen -= 1
                #
            else: # それ以外の選択されなかったジョブはデータ転送しない
                delay, deadline_length, job_size, done = self.job_queue[i].step(served=False,time=self.time)
            if done == True: # 転送が完了したジョブの処理
                # reward_i = math.exp(-self.mu*delay) # 終了したジョブの即時報酬を計算する
                reward_i = job_size * math.exp(-self.mu*delay) # 終了したジョブの即時報酬を計算する ジョブ長を考慮した改良版
                # reward_i = delay # 終了したジョブの即時報酬を計算する
                completed_job_size_i = job_size
                completed_job_num_i = 1
                done_idx = i # 待ち行列から転送が完了したジョブ番号を覚えておく
            else:
                reward_i = 0
                completed_job_size_i = 0
                completed_job_num_i = 0
            reward += reward_i
            completed_job_size += completed_job_size_i
            completed_job_num += completed_job_num_i
        if done_idx != -1:
            del self.job_queue[done_idx] # 待ち行列から転送が完了したジョブを取り除く
        self.qlen = max(self.qlen,0)
        #
        nact_job_queue = []
        if (len(self.job_queue) >= self.nact): # 待ち行列にいるジョブ数がnact以上の場合は先頭からnactを取ってくる
            nact_job_queue = copy.copy(self.job_queue[0:self.nact])
        else: # 待ち行列にいるジョブ数がnactより小さい場合はその分をダミーデータで埋める
            nact_job_queue = copy.copy(self.job_queue)
            dummy_job = Job()
            dummy_job.turn_dummy()
            for i in range(self.nact-len(self.job_queue)):
                nact_job_queue.append(dummy_job)
        #
        list_arriving_jobs = self.jobset.step() # このステップにおいて新しく到着するジョブ
        self.job_queue.extend(list_arriving_jobs) # このステップにおいて新しく到着したジョブを待ち行列に追加
        arriving_job_size,_ = get_list_job_info(list_arriving_jobs)
        self.qlen += arriving_job_size
        # #
        # #
        # print(f"!!! time:{self.time}, index:{index}")
        # self.jobset.show_config()
        # #
        terminated = False
        truncated = False
        # # #
        observation, _, _, _ = make_observation(list_jobs = nact_job_queue, time = self.time)
        info =  {'time': self.time, 'completed_job_size': completed_job_size,'completed_job_num': completed_job_num, 'qlen': self.qlen} 
        # #
        # 2023-07-19　これまでは時計を1ステップ進めてからobservationを計算していたが、そうすべきでないと考えたので、順番を入れ替えた
        # #
        self.time += 1 # 時計を1ステップ進める
        #
        # # # ここのエピソード終了判定はRLの場合は見直す必要がある。特に、Traceの場合？
        # # # すなわち、まだジョブが残っている限り、延長する。
        if self.time >= self.tmax-1:
            if len(self.job_queue) <= 0:
                terminated = True
                truncated = True
            else: # busy periodを超えてもジョブが残っている場合
                # # 負の報酬を計算　まだ残っているジョブの大きさ
                # _, remain_job_size = get_list_job_info(self.job_queue)
                # reward -= remain_job_size
                # # 負の報酬を計算　デッドラインを超えていたらそれも加算
                # for i in range(len(self.job_queue)): # 待ち行列にいるジョブについて処理をする
                #     ######################  次の行でjobクラスのstepメソッドでは時計は進まないので問題なし
                #     delay, deadline_length, job_size, done = self.job_queue[i].step(served=False,time=self.time)
                #     reward_i = job_size * (1-math.exp(-self.mu*delay)) # 終了したジョブの即時報酬を計算する ジョブ長を考慮した改良版
                #     reward -= reward_i
                # #
                # # print(f"tmax is extend. Reward: {reward}, {len(self.job_queue)} jobs are still in the queue.")
                if self.time >= 2*self.tmax: # busy periodの2倍を超えてもジョブが残っていれば、episodeを終了させる
                    terminated = True
                    truncated = True

        #
        return observation, reward, terminated, truncated, info

    # def reset(
    #     self,
    #     *,
    #     seed: Optional[int] = None,
    #     options: Optional[dict] = None,
    # ) -> Tuple[ObsType, dict]:
    def reset(self,
              *,
              seed = None,
              options = None):
        """Resets the environment to an initial state and returns the initial observation.

        This method can reset the environment's random number generator(s) if ``seed`` is an integer or
        if the environment has not yet initialized a random number generator.
        If the environment already has a random number generator and :meth:`reset` is called with ``seed=None``,
        the RNG should not be reset. Moreover, :meth:`reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG.
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)


        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            #
            self.seed = seed
        #
        # JobSetをはじめに戻す
        self.jobset.reset()
        self.time = 0
        self.tmax = self.jobset.get_tmax()
        self.qlen = 0
        #
        #
        self.job_queue = [] # 現在、処理を待っているジョブの待ち行列
        nact_job_queue = []
        if (len(self.job_queue) >= self.nact): # 待ち行列にいるジョブ数がnact以上の場合は先頭からnactを取ってくる
            nact_job_queue = copy.copy(self.job_queue[0:self.nact])
        else: # 待ち行列にいるジョブ数がnactより小さい場合はその分をダミーデータで埋める
            nact_job_queue = copy.copy(self.job_queue)
            dummy_job = Job()
            dummy_job.turn_dummy()
            for i in range(self.nact-len(self.job_queue)):
                nact_job_queue.append(dummy_job)
        observation, _, _, _ = make_observation(list_jobs = nact_job_queue, time = self.time)
        #
        info =  {'k1': 1, 'k2': 2, 'k3': 3} # dummy
        # #
        # #
        # self.jobset.show_config()
        # #
        # #
        return observation, info

    # def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
    def render(self):
        """Compute the render frames as specified by render_mode attribute during initialization of the environment.

        The set of supported modes varies per environment. (And some
        third-party environments may not support rendering at all.)
        By convention, if render_mode is:

        - None (default): no render is computed.
        - human: render return None.
          The environment is continuously rendered in the current display or terminal. Usually for human consumption.
        - rgb_array: return a single frame representing the current state of the environment.
          A frame is a numpy.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
        - rgb_array_list: return a list of frames representing the states of the environment since the last reset.
          Each frame is a numpy.ndarray with shape (x, y, 3), as with `rgb_array`.
        - ansi: Return a strings (str) or StringIO.StringIO containing a
          terminal-style text representation for each time step.
          The text can include newlines and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render_modes' key includes
            the list of supported modes. It's recommended to call super()
            in implementations to use the functionality of this method.
        """
        print("------------------------------------------------------------")
        #
        # print(self.job_queue)
        # self.jobset.show()
        num_jobs = len(self.job_queue)
        print(f"time: {self.time}")
        print(f"num of jobs: {num_jobs}")
        print(f"qlen: {self.qlen}")
        # print(f"{self.time},{num_jobs},{self.qlen}")
        show_job_queue(self.job_queue)
        
    def get_status(self):
        # num_jobs = len(self.job_queue)
        observation, num_obs_elem, num_jobs, job_size_remain = make_observation(list_jobs = self.job_queue, time = self.time)        
        return self.time, num_jobs, job_size_remain

    def get_nact(self):
        return self.nact

    def get_num_obs_elem(self):
        return self.num_obs_elem

    def show_status(self):
        observation, num_obs_elem, num_jobs, job_size_remain = make_observation(list_jobs = self.job_queue, time = self.time)        
        print(f"# time:{self.time}, num_jobs:{num_jobs}, job_size_remain:{job_size_remain}")
        for item in self.job_queue:
            arrival_time, deadline_length, deadline, job_size, job_size_remain = item.get_job_info()
            print(f"# arrival_time:{arrival_time}, deadline_length:{deadline_length}, deadline:{deadline}, job_size:{job_size}, job_size_remain:{job_size_remain}")


    def sample(self,
               t0, t1):
        """
        Busy periodのジョブセットを取り出す
        Args: starting time t0, end time t1-1
        Returns: a subset of the jobset
        """
        return self.jobset.sample(t0,t1)
