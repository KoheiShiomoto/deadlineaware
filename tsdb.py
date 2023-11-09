import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
from operator import itemgetter

class EvalBusyPeriod():
    def __init__(self,
                 ofileName_base):
        self.ofileName_base = ofileName_base
        self.is_busy = False # if we are in a busy period, True. Otherwise False
        self.list_busy_period = []
        self.sum_reward = 0
        self.sum_completed_job_size = 0
        self.sum_completed_job_num = 0
        return
    
    def step(self,
             reward,
             info):
        time = info["time"]
        completed_job_size = info["completed_job_size"]
        completed_job_num = info["completed_job_num"]
        qlen = info["qlen"]
        # print(f"time:{time}, job_size:{completed_job_size}, job_num:{completed_job_num}, qlen:{qlen}")
        self.sum_reward += reward
        self.sum_completed_job_size += completed_job_size
        self.sum_completed_job_num += completed_job_num
        #
        if self.is_busy == False and qlen > 0 : # busy periodの開始
            self.is_busy = True
            self.time_start_bp = time
        elif self.is_busy == True and qlen == 0 : # busy periodの終了 このときbusy periodを評価
            perf = self.sum_reward / self.sum_completed_job_size
            self.list_busy_period.append([self.time_start_bp, time, perf, self.sum_reward, self.sum_completed_job_size, self.sum_completed_job_num])
            self.sum_reward = 0
            self.sum_completed_job_size = 0
            self.sum_completed_job_num = 0
            #
            self.is_busy = False
             
    def to_csv(self):
        self.df = pd.DataFrame(self.list_busy_period,
                  columns=['time_start', 'time_end', 'perf', 'reward', 'duty','num_job'])

        self.df['bp_len'] = self.df['time_end'] - self.df['time_start']
        # print(self.df)
        self.df.to_csv(self.ofileName_base+"_eval_busy_period.csv")

    #
    # 2023-09-07
    # def from_csv(self,
    #              perf_thresh=1.0):
    #     """
    #     csvファイルからperfが閾値以下のBusy periodのみを読み出し、Busy periodのリストを返す
    #     Args: perf_thresh: threshold for perf
    #     Returns: None
    #     """
    #     df = pd.read_csv(self.ofileName_base+"_eval_busy_period.csv")
    #     self.df = df[df['perf'] <= perf_thresh]
    #     t0_list = self.df['time_start'].values.tolist()
    #     t1_list = self.df['time_end'].values.tolist()
    #     bp_list = []
    #     for t0, t1 in zip(t0_list, t1_list):
    #         bp_list.append([t0, t1])
    #     return bp_list
    def from_csv(self,
                 perf_thresh=1.0,
                 sample_size=1000000):
        """
        csvファイルからperfが閾値以下のBusy periodのみを読み出し、Busy periodのリストを返す
        Busy period数は固定する
        Args: 
           perf_thresh: perf threshold to sample busy periods
           sample_size: number of sampled busy periods
        Returns: None
        """
        df = pd.read_csv(self.ofileName_base+"_eval_busy_period.csv")
        #
        # 2023-10-15 perfの昇順で（性能の悪い順番で）DFをソート
        df = df.sort_values('perf', ascending=True) # perfの昇順でDFをソート
        # 2023-10-15
        #
        self.df = df[df['perf'] <= perf_thresh]
        t0_list_base = self.df['time_start'].values.tolist()
        t1_list_base = self.df['time_end'].values.tolist()
        perf_list_base = self.df['perf'].values.tolist()
        reward_list_base = self.df['reward'].values.tolist()
        #
        # 2023-09-07
        data_size = len(t0_list_base)
        if sample_size >= 1000000:
            sample_size = data_size
        assert data_size >= sample_size
        shuffled_idx = np.random.choice(np.arange(data_size), sample_size, replace=False)
        t0_list = itemgetter(*shuffled_idx)(t0_list_base)
        t1_list = itemgetter(*shuffled_idx)(t1_list_base)
        perf_list = itemgetter(*shuffled_idx)(perf_list_base)
        reward_list = itemgetter(*shuffled_idx)(reward_list_base)
        duty_list = itemgetter(*shuffled_idx)(duty_list_base)
        num_job_list = itemgetter(*shuffled_idx)(num_job_list_base)
        # 2023-09-07
        #
        bp_list = []
        # for t0, t1 in zip(t0_list, t1_list):
        for t0, t1, perf, reward, duty, num_job in zip(t0_list, t1_list, perf_list, reward_list, duty_list, num_job_list):
            # bp_list.append([t0, t1])
            bp_list.append([t0, t1, perf, reward, duty, num_job])
        return bp_list
    
    def plot(self):
        x = self.df['bp_len'].to_numpy()
        y = self.df['perf'].to_numpy()
        #
        fig = plt.figure()
        ax = fig.add_subplot(111)
        H = ax.scatter(x, y)
        ax.set_xlabel('busy period length')
        ax.set_ylabel('perf')
        plt.savefig(self.ofileName_base+"_eval_busy_period_scatter.pdf")
        plt.show()
        #
        x = self.df['bp_len'].to_numpy()
        y = self.df['num_job'].to_numpy()
        z = self.df['perf'].to_numpy()
        #
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        H = ax.scatter(x, y, z)
        ax.set_xlabel('busy period length')
        ax.set_ylabel('number of jobs')
        ax.set_zlabel('perf')
        plt.savefig(self.ofileName_base+"_eval_busy_period_scatter3d.pdf")
        plt.show()
        #
        x = self.df['perf'].to_numpy()
        y = self.df['duty'].to_numpy()
        #
        fig = plt.figure()
        ax = fig.add_subplot(111)
        H = ax.hist(x, bins=10)
        ax.set_xlabel('perf')
        ax.set_ylabel('counts')
        plt.savefig(self.ofileName_base+"_eval_busy_period_hstgrm.pdf")
        plt.show()
        #
        fig = plt.figure()
        ax = fig.add_subplot(111)
        H = ax.hist2d(x,y, bins=[40,10], cmap=cm.jet)
        ax.set_xlabel('perf')
        ax.set_ylabel('duty')
        fig.colorbar(H[3],ax=ax)
        plt.savefig(self.ofileName_base+"_eval_busy_period_heatmap.pdf")
        plt.show()

class TsDatabase():
    def __init__(self,
                 ofileName_base):
        self.key_list = []
        self.data = {}
        self.ofileName_base = ofileName_base
        return
    
    def add(self,key_value_lists):
        for item in key_value_lists:
            key, value = item
            if key in self.key_list:
                self.data[key].append(value)
            else:
                self.key_list.append(key)
                self.data[key]=[]
                self.data[key].append(value)
        
    def plot(self):
        key_list = self.key_list.remove("time")
        for key in self.key_list:
            plt.clf()
            plt.plot(self.data["time"], self.data[key], label=key)
            plt.xlabel("time") # x軸のラベル
            plt.ylabel(key) # y軸のラベル
            plt.title(f"{key} as a function of time")
            plt.legend()
            plt.savefig(self.ofileName_base+"_"+key+".pdf")
            plt.show()

    def to_csv(self):
        df = pd.DataFrame(self.data)
        print(df)
        df.to_csv(self.ofileName_base+"_tsdata.csv")


# ,AccumReward,time,num_jobs,qlen
# 0,0.0,1,0,0.0
# 1,0.0,2,7,10.0
# 2,0.0,3,7,9.0
class JoinTsDatabase():
    def __init__(self,
                 ofileName_base):
        self.ofileName_base = ofileName_base
        self.df_AccumReward = pd.DataFrame()
        self.df_num_jobs = pd.DataFrame()
        self.df_qlen = pd.DataFrame()
        return
    
    # key_value_lists includes the list of column_name and file_name, e.g., ('EDF','odata_EDF_traceK1_tsdata.csv')
    def merge(self,key_value_lists):
        for item in key_value_lists:
            column_name, file_name = item 
            print(f"column_name:{column_name}, file_name:{file_name}")
            # sub_column is as follows
            # ,AccumReward,time,num_jobs,qlen
            # 0,0.0,1,0,0.0
            # 1,0.0,2,7,10.0
            # 2,0.0,3,7,9.0
            df_AccumReward_tmp = pd.read_csv(file_name, names=(column_name,'time'), header=0, usecols=[2,1])
            if self.df_AccumReward.empty:
                self.df_AccumReward = df_AccumReward_tmp.set_index('time')
            else:
                # self.df_AccumReward.merge(df_AccumReward_tmp, on='time')
                self.df_AccumReward = self.df_AccumReward.merge(df_AccumReward_tmp, on='time')
            #
            df_num_jobs_tmp = pd.read_csv(file_name, names=('time',column_name), header=0, usecols=[2,3])
            if self.df_num_jobs.empty:
                self.df_num_jobs = df_num_jobs_tmp.set_index('time')
            else:
                self.df_num_jobs = self.df_num_jobs.merge(df_num_jobs_tmp, on='time')
            #
            df_qlen_tmp = pd.read_csv(file_name, names=('time',column_name), header=0, usecols=[2,4])
            if self.df_qlen.empty:
                self.df_qlen = df_qlen_tmp.set_index('time')
            else:
                self.df_qlen = self.df_qlen.merge(df_qlen_tmp, on='time')
        print("AccumReward")
        print(self.df_AccumReward)
        print("num_jobs")
        print(self.df_num_jobs)
        print("qlen")
        print(self.df_qlen)
        
    def plot(self):
        plt.clf()
        self.df_AccumReward.plot(title='AccumReward', x='time')
        plt.savefig(self.ofileName_base+"_mergedTsData_AccumReward.pdf")
        plt.show()
        plt.clf()
        self.df_num_jobs.plot(title='Number of jobs in queue', x='time')
        plt.savefig(self.ofileName_base+"_mergedTsData_num_jobs.pdf")
        plt.show()
        plt.clf()
        self.df_qlen.plot(title='Data size in job queue', x='time')
        plt.savefig(self.ofileName_base+"_mergedTsData_qlen.pdf")
        plt.show()


    def to_csv(self):
        self.df_AccumReward.to_csv(self.ofileName_base+"_mergedTsData_AccumReward.csv")
        self.df_num_jobs.to_csv(self.ofileName_base+"_mergedTsData_num_jobs.csv")
        self.df_qlen.to_csv(self.ofileName_base+"_mergedTsData_qlen.csv")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pj','--pjName',default='pjName', help='Base of output file name (default: pjName)')  
    #
    #

    args = parser.parse_args()
    pjName = args.pjName
    ofileName_base = "output/odata_"+pjName

    dataBase = JoinTsDatabase(ofileName_base = ofileName_base+'_self_e1000')
    kv_list = [('Actor-Critic','output/odata_AC_lr05_self_alp15_ep50000_tsdata.csv'),
               ('EDF','output/odata_EDF_self_alp15_ep1000_tsdata.csv'),
               ('FCFS','output/odata_FCFS_self_alp15_ep1000_tsdata.csv')]
    # kv_list = [('Actor-Critic','output/odata_AC_lr05_self_alp15_ep1000_tsdata.csv'),
    #            ('EDF','output/odata_EDF_self_alp15_ep1000_tsdata.csv'),
    #            ('FCFS','output/odata_FCFS_self_alp15_ep1000_tsdata.csv')]
    dataBase.merge(kv_list)
    dataBase.plot()
    dataBase.to_csv()    

    dataBase = JoinTsDatabase(ofileName_base = ofileName_base+'_K1')
    kv_list = [('Actor-Critic','output/odata_AC_traceK1_lr05_ep50000_tsdata.csv'),
               ('EDF','output/odata_EDF_traceK1_tsdata.csv')]
    # kv_list = [('Actor-Critic','output/odata_AC_traceK1_lr05_ep10000_tsdata.csv'),
    #            ('EDF','output/odata_EDF_traceK1_tsdata.csv'),
    #            ('FCFS','output/odata_FCFS_traceK1_tsdata.csv')]
    dataBase.merge(kv_list)
    dataBase.plot()
    dataBase.to_csv()    

    dataBase = JoinTsDatabase(ofileName_base = ofileName_base+'_K2')
    kv_list = [('Actor-Critic','output/odata_AC_traceK2_lr05_ep50000_tsdata.csv'),
               ('EDF','output/odata_EDF_traceK2_tsdata.csv')]
    # kv_list = [('Actor-Critic','output/odata_AC_traceK2_lr05_ep10000_tsdata.csv'),
    #            ('EDF','output/odata_EDF_traceK2_tsdata.csv'),
    #            ('FCFS','output/odata_FCFS_traceK2_tsdata.csv')]
    dataBase.merge(kv_list)
    dataBase.plot()
    dataBase.to_csv()    

    dataBase = JoinTsDatabase(ofileName_base = ofileName_base+'_K4')
    kv_list = [('Actor-Critic','output/odata_AC_traceK4_lr05_ep50000_tsdata.csv'),
               ('EDF','output/odata_EDF_traceK4_tsdata.csv')]
    # kv_list = [('Actor-Critic','output/odata_AC_traceK4_lr05_ep10000_tsdata.csv'),
    #            ('EDF','output/odata_EDF_traceK4_tsdata.csv'),
    #            ('FCFS','output/odata_FCFS_traceK4_tsdata.csv')]
    dataBase.merge(kv_list)
    dataBase.plot()
    dataBase.to_csv()    
