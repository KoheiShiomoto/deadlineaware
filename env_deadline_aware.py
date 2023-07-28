import numpy as np
import math
import random
import copy
import csv

from common.tools import decomment

class Job():
    def __init__(self,
                 arrival_time = -1,
                 deadline_length = 0,
                 job_size = 0): # 初期値はダミーデータ[arrival_time, deadline_length, job_size] = [-1,0,0]
        """
        Args:

        Returns:
        """
        self.arrival_time = arrival_time # 到着時刻
        self.deadline_length = deadline_length # デッドラインの長さ
        self.deadline = arrival_time+deadline_length # デッドライン時間（絶対時刻）
        self.job_size = job_size # ジョブサイズ（到着時のもの）
        self.job_size_remain = job_size # ジョブの残りのサイズ

    def turn_dummy(self): # ダミーデータ[arrival_time, deadline_length, job_size] = [-1,0,0]　にセットする
        self.arrival_time = -1 # 到着時刻
        self.deadline_length = 0 # デッドラインの長さ
        self.deadline = -1 # デッドライン時間（絶対時刻）
        self.job_size = 0 # ジョブサイズ（到着時のもの）
        self.job_size_remain = 0 # ジョブの残りのサイズ
        return

    def show(self):
        print(f"# arrival_time:{arrival_time}, deadline_length:{self.deadline_length}, deadline:{self.deadline}, job_size:{self.job_size}, job_size_remain:{self.job_size_remain}")
        return

    def get_job_info(self):
        return self.arrival_time, self.deadline_length, self.deadline, self.job_size, self.job_size_remain
    
    def step(self,
             served,
             time):
        """
        1ステップ進めるときの更新処理
        servedはbool型で選択されたジョブの場合はTrueでそれ以外はFalse
        delayとdoneを返す
        Args:

        Returns:
        """
        if served == True:
            self.job_size_remain -= 1.0
        if self.job_size_remain <= 0:
            delay = max(time-self.deadline,0.0)
            done = True
        else:
            delay = 0
            done = False
        return delay, self.deadline_length, self.job_size, done


def show_job_queue(job_queue):
    for item in job_queue:
        arrival_time, deadline_length, deadline, job_size, job_size_remain = item.get_job_info()
        print(f"# arrival_time:{arrival_time}, deadline_length:{deadline_length}, deadline:{deadline}, job_size:{job_size}, job_size_remain:{job_size_remain}")
    return 

def get_list_job_info(list_jobs):
    sum_job_size = 0
    sum_job_size_remain = 0
    for item in list_jobs:
        _, _, _, job_size, job_size_remain = item.get_job_info()
        sum_job_size += job_size
        sum_job_size_remain += job_size_remain
    return sum_job_size, sum_job_size_remain


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
    l = [0,0] # [deadline_remain, job_size_remain] # subset1 attributes
    for item in observation:
        arrival_time, deadline_length, deadline_remain, job_size, job_size_remain = item[0], item[1], item[2], item[3], item[4] 
        if job_size == 0:
            deadline_remain = 0
            job_size_remain = 0
        deadline_remain = clip(deadline_remain)
        l = [deadline_remain, job_size_remain] # subset1 attributes most useful pattern? 2
            
        num_state_elem = len(l)
        state.append(l)
    return state, num_state_elem

def make_observation(list_jobs, time):
    observation = []
    l = [0,0,0,0,0] # [arrival_time, deadline_length, deadline_remain, job_size, job_size_remain] # full attributes
    num_obs_elem = 0
    num_jobs = 0
    sum_job_size_remain = 0
    for item in list_jobs:
        arrival_time, deadline_length, deadline, job_size, job_size_remain = item.get_job_info()
        deadline_remain = deadline-time
        l = [arrival_time, deadline_length, deadline_remain, job_size, job_size_remain] # full attributes
        num_obs_elem = len(l)
        num_jobs += 1
        sum_job_size_remain += job_size_remain
        observation.append(l)
    return observation, num_obs_elem, num_jobs, sum_job_size_remain
    

class ArrivalProcess():
    def __init__(self,
                 seed = 0,
                 lambda_ = 0.08,
                 alpha = 2.0,
                 beta = 10.0):
        self.seed = seed # 乱数の種を変更できるようにする
        np.random.seed(seed) # ポアソン分布疑似乱数モジュールを持つnpのシードの設定はこちら
        random.seed(seed) # これは不要か？
        self.lambda_ = lambda_ # ポアソン過程のパラメータ（1スロットに到着するジョブ数を決める）
        self.alpha = alpha # 平均デッドライン長 ジョブサイズの係数倍
        self.beta = beta # 平均ジョブサイズ

    def genJob(self,
               time):
        """
        1スロットに生成したジョブのリストを返す．
        Args:

        Returns:
        """
        slot = 1
        n = np.random.poisson(self.lambda_,slot)
        list_jobs = []
        for i in range(int(n)):
            # job_size = self.beta # ジョブ長は乱数で決める
            job_size = int(np.random.exponential(1.0*self.beta)) # ジョブ長は乱数で決める
            deadline_length = int(job_size * random.uniform(1, self.alpha)) # ジョブ長が決まったら、デッドラインをk倍(k>1)とする kは乱数でもよい
            job = Job(arrival_time=time,
                      deadline_length=deadline_length,
                      job_size=job_size)
            list_jobs.append(job)
        return list_jobs

class JobSet():
    def __init__(self,
                 tmax,
                 seed = 0,
                 isStream = True,
                 lambda_ = 0.08,
                 alpha = 2.0,
                 beta = 10.0):
        """
        指定されたパラメータでジョブの到着予定表を生成する
        時刻0で最初のジョブが到着したパターンを生成する
        Busy Periodが終了するか、時間切れtmaxに到達するまでのパターンを作成する
        Args:

        Returns:
        """
        self.isStream = isStream
        self.lambda_ = lambda_ # ポアソン過程のパラメータ（1スロットに到着するジョブ数を決める）
        self.alpha = alpha # 平均デッドライン長 ジョブサイズの係数倍
        self.beta = beta # 平均ジョブサイズ
        #
        self.tmax = tmax
        self.time = 0
        self.seed = seed
        self.calender = []
        self.arrival_process = ArrivalProcess(seed = self.seed,
                                              lambda_ = self.lambda_,
                                              alpha = self.alpha,
                                              beta = self.beta)
        self.configure() # 
        #
        self.time = 0
        self.qlen = 0

    def get_tmax(self):
        return self.tmax
    

    def configure(self):
        """
        時刻0で最初のジョブが到着したパターンを生成する
        Busy Periodが終了するか、時間切れtmaxに到達するまでのパターンを作成する
        仕事量保存則が成り立つ前提でFIFOでサービスした場合のBusy Periodを作成
        Args:

        Returns:
        """
        isFirst = True
        time = 0
        self.qlen = 0
        while time < self.tmax:
            list_arriving_jobs = self.arrival_process.genJob(time=time)
            if len(list_arriving_jobs) != 0 and isFirst == True:
                isFirst = False
            if isFirst == False:
                arriving_job_size,_ = get_list_job_info(list_arriving_jobs)
                self.qlen += arriving_job_size-1
                self.qlen = max(self.qlen,0)
                # if ((self.qlen == 0) and (self.isStream == False)) or time == self.tmax-1:
                if (self.qlen == 0) and (self.isStream == False) :
                    self.tmax = time+1
                    break
                self.calender.append(list_arriving_jobs)
                time += 1
        return

    def step(self):
        """
        1ステップ進める。時計も進める。
        Args:

        Returns:
        """
        if self.time < self.tmax:
            list_arriving_jobs = self.calender[self.time]
        else:
            list_arriving_jobs = []
        self.time += 1
        return list_arriving_jobs
    


    def reset(self):
        """
        同じサンプルパスで最初からジョブを生成する
        Args:

        Returns:
         """
        self.time = 0
        #
        self.calender = []
        self.arrival_process = ArrivalProcess(seed = self.seed,
                                              lambda_ = self.lambda_,
                                              alpha = self.alpha,
                                              beta = self.beta)
        self.configure() # 
        #
        return

    def show_config(self):
        print("----------------------------")
        print(f"tmax={self.tmax}, time={self.time}")
        print("calender")
        for time, list_arricing_jobs in enumerate(self.calender):
            print(f"time:{time}")
            for item in list_arricing_jobs:
                arrival_time, deadline_length, deadline, job_size, job_size_remain = item.get_job_info()
                print(f"# arrival_time:{arrival_time}, deadline_length:{deadline_length}, deadline:{deadline}, job_size:{job_size}, job_size_remain:{job_size_remain}")
            
        
        

class TraceJobSet():
    def __init__(self,
                 tmax,
                 f_name):
        """
        指定されたファイルからジョブのパターンを読み込む
        Args:

        Returns:
        """
        #
        self.tmax = tmax
        self.f_name = f_name
        self.calender = []
        for i in range(self.tmax):
            self.calender.append([])
        self.configure() # 

    def get_tmax(self):
        return self.tmax
       
    def configure(self):
        """
        Args:

        Returns:
        """
        # # time, job_size, deadline_length
        # 0,2,3
        # ...
        with open(self.f_name) as f:
            reader = csv.reader(decomment(f))
            for row in reader:
                time = int(row[0])
                job_size = int(row[1])
                deadline_length = int(row[2])
                # print(row[0],row[1],row[2])
                job = Job(arrival_time = time,
                          deadline_length = deadline_length,
                          job_size = job_size)
                self.calender[time].append(job)
        return

    def step(self):
        """
        1ステップ進める。時計も進める。
        Args:

        Returns:
        """
        if self.time < self.tmax:
            list_arriving_jobs = self.calender[self.time]
        else:
            list_arriving_jobs = []
        self.time += 1
        return list_arriving_jobs

    def reset(self):
        """
        同じサンプルパスで最初からジョブを生成する
        Args:

        Returns:
         """
        self.time = 0
        #
        self.calender = []
        for t in range(self.tmax):
            self.calender.append([])
        self.configure() # 
        # 
        # 同じ時間に到着するジョブの並び順番をランダムに変える
        for t in range(self.tmax):
            random.shuffle(self.calender[t])
        #
        return
    
    def show_config(self):
        print("----------------------------")
        print(f"tmax={self.tmax}, time={self.time}")
        print("calender")
        for time, list_arricing_jobs in enumerate(self.calender):
            print(f"time:{time}")
            for item in list_arricing_jobs:
                arrival_time, deadline_length, deadline, job_size, job_size_remain = item.get_job_info()
                print(f"# arrival_time:{arrival_time}, deadline_length:{deadline_length}, deadline:{deadline}, job_size:{job_size}, job_size_remain:{job_size_remain}")

class EnvDeadlineAware():
    def __init__(self,
                 isStream = True,
                 isTraceJobSet = False,
                 f_name = "data_jobset.txt",
                 tmax = 300,
                 seed = 0,
                 nact = 8,
                 mu = 1.0,
                 lambda_ = 0.08,
                 beta = 10.0,
                 alpha = 2.0):
        """
        Args:

        Returns:
        """
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
        #
        self.job_queue = [] # 現在、処理を待っているジョブの待ち行列
        # 最初にここでJobSetを作っておく

        if isTraceJobSet == True:
            self.jobset = TraceJobSet(tmax = self.tmax,
                                      f_name = self.f_name)
        else:
            self.jobset = JobSet(tmax = self.tmax,
                                 seed = self.seed,
                                 isStream = self.isStream,
                                 lambda_ = self.lambda_,
                                 alpha = self.alpha,
                                 beta = self.beta)
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
        
    def step(self, action):
        #
        index = int(action)
        reward = 0
        completed_job_size = 0
        completed_job_num = 0
        done_idx = -1 # 待ち行列から転送が完了したジョブ番号を覚えておく
        for i in range(len(self.job_queue)): # 待ち行列にいるジョブについて処理をする
            if i == index: # indexで示された1個のジョブを選択してデータ転送する
                delay, deadline_length, job_size, done = self.job_queue[i].step(served=True,time=self.time)
                self.qlen -= 1
            else: # それ以外の選択されなかったジョブはデータ転送しない
                delay, deadline_length, job_size, done = self.job_queue[i].step(served=False,time=self.time)
            if done == True: # 転送が完了したジョブの処理
                reward_i = job_size * math.exp(-self.mu*delay) # 終了したジョブの即時報酬を計算する ジョブ長を考慮した改良版
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
        terminated = False
        truncated = False
        observation, _, _, _ = make_observation(list_jobs = nact_job_queue, time = self.time)
        info =  {'time': self.time, 'completed_job_size': completed_job_size,'completed_job_num': completed_job_num, 'qlen': self.qlen} 
        self.time += 1 # 時計を1ステップ進める
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
                #     delay, deadline_length, job_size, done = self.job_queue[i].step(served=False,time=self.time)
                #     reward_i = job_size * (1-math.exp(-self.mu*delay)) # 終了したジョブの即時報酬を計算する ジョブ長を考慮した改良版
                #     reward -= reward_i
                # #
                if self.time >= 2*self.tmax: # busy periodの2倍を超えてもジョブが残っていれば、episodeを終了させる
                    terminated = True
                    truncated = True

        #
        return observation, reward, terminated, truncated, info

    def reset(self,
              *,
              seed = None,
              options = None):
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
        info =  {'k1': 1, 'k2': 2, 'k3': 3} # dummy
        return observation, info

    def render(self):
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
