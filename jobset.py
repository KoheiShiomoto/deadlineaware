import numpy as np
import random
import csv

from common.tools import decomment

def get_list_job_info(list_jobs):
    sum_job_size = 0
    sum_job_size_remain = 0
    for item in list_jobs:
        _, _, _, job_size, job_size_remain = item.get_job_info()
        sum_job_size += job_size
        sum_job_size_remain += job_size_remain
    return sum_job_size, sum_job_size_remain

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
        # self.deadline = arrival_time+deadline_length # デッドライン時間（絶対時刻）
        # 2023-08-09
        self.deadline = arrival_time+deadline_length+1 # デッドライン時間（絶対時刻）に1を加える　必ず1スロットは待つため 2023-07-19の修正への対応
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
                 tmax):
        self.time = 0
        self.tmax = tmax
        self.qlen = 0
        self.calender = []

    def step(self):
        """
        1ステップ進める。時計も進める。
        Args: None
        Returns: a list of arriving jobs
        """
        if self.time < self.tmax:
            list_arriving_jobs = self.calender[self.time]
        else:
            list_arriving_jobs = []
        self.time += 1
        return list_arriving_jobs

    def set_calender(self,
                     calender):
        self.calender = calender

    def get_calender(self):
        return self.calender
        
    def sample(self,
               t0, t1):
        """
        Busy periodのジョブセットを取り出す
        Args: starting time t0, end time t1-1
        Returns: a subset of the jobset
        """
        subset = JobSet(t1-t0)
        subset_calender = []
        for t in range(t0,t1):
            list_arriving_jobs = self.calender[t]
            subset_calender.append(list_arriving_jobs)
        subset.set_calender(subset_calender)
        return subset

    def reset(self):
        """
        同じサンプルパスで最初からジョブを生成する
        Args: None
        Returns: None
         """
        self.time = 0
        self.calender = []
        return

    def get_tmax(self):
        return self.tmax

    def show_config(self):
        print("----------------------------")
        print(f"tmax={self.tmax}, time={self.time}")
        print("calender")
        for time, list_arricing_jobs in enumerate(self.calender):
            print(f"time:{time}")
            for item in list_arricing_jobs:
                arrival_time, deadline_length, deadline, job_size, job_size_remain = item.get_job_info()
                print(f"# arrival_time:{arrival_time}, deadline_length:{deadline_length}, deadline:{deadline}, job_size:{job_size}, job_size_remain:{job_size_remain}")


class SelfJobSet(JobSet):
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
        Args: None
        Returns: None
        """
        super().__init__(tmax = tmax)
        #
        self.seed = seed
        self.isStream = isStream
        self.lambda_ = lambda_ # ポアソン過程のパラメータ（1スロットに到着するジョブ数を決める）
        self.alpha = alpha # 平均デッドライン長 ジョブサイズの係数倍
        self.beta = beta # 平均ジョブサイズ
        self.arrival_process = ArrivalProcess(seed = self.seed,
                                              lambda_ = self.lambda_,
                                              alpha = self.alpha,
                                              beta = self.beta)
        self.configure()

    def configure(self):
        """
        時刻0で最初のジョブが到着したパターンを生成する
        Busy Periodが終了するか、時間切れtmaxに到達するまでのパターンを作成する
        仕事量保存則が成り立つ前提でFIFOでサービスした場合のBusy Periodを作成
        Args: None
        Returns: None
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
        
    def reset(self):
        """
        同じサンプルパスで最初からジョブを生成する
        Args: None
        Returns: None
         """
        super().reset()
        self.arrival_process = ArrivalProcess(seed = self.seed,
                                              lambda_ = self.lambda_,
                                              alpha = self.alpha,
                                              beta = self.beta)
        self.configure()
        return


class TraceJobSet(JobSet):
    def __init__(self,
                 tmax,
                 f_name):
        """
        指定されたファイルからジョブのパターンを読み込む
        Args:
        Returns:
        """
        #
        super().__init__(tmax = tmax)
        self.f_name = f_name
        self.configure()
       
    def configure(self):
        """
        Args:
        Returns:
        """
        # # time, job_size, deadline_length
        # 0,2,3
        # ...
        for t in range(self.tmax):
            self.calender.append([])
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

    def reset(self):
        super().reset()
        self.configure()
        # 同じ時間に到着するジョブの並び順番をランダムに変える
        #  -> 固定パターンのままだと過学習が起きて性能が劣化したと記憶しているが、あとで試すと劣化しなかった?
        for t in range(self.tmax):
            random.shuffle(self.calender[t])
        #
        return
    
class DrillJobSet(JobSet):

    def __init__(self,
                  jobset,
                  list_bp):
        self.jobset_dict = []
        for i, bp in enumerate(list_bp):
            t0, t1 = bp[0], bp[1]
            self.jobset_dict.append(jobset.sample(t0,t1))
        idx = random.randrange(len(self.jobset_dict)) # 問題集からランダムにジョブセットを選ぶ
        self.configure(idx = idx)
        
    def configure(self,
                  idx):
        """
        Args:
        Returns:
        """
        #
        jobset = self.jobset_dict[idx]
        tmax = jobset.get_tmax()
        calender = jobset.get_calender()
        super().__init__(tmax = tmax)
        self.set_calender(calender)

    def reset(self):
        super().reset()
        idx = random.randrange(len(self.jobset_dict)) # 問題集からランダムにジョブセットを選ぶ
        # print(f"# jobset idx:{idx}")
        self.configure(idx = idx)
        return

    def show_drill_config(self):
        num_jobsets = len(self.jobset_dict)
        print(f"# number of jobsets:{num_jobsets}")
        # for i in range(num_jobsets):
        #     self.jobset_dict[i].show_conifg()

