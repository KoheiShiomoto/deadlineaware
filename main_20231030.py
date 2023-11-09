import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
import argparse
import pickle

from jobset import get_list_job_info, Job, SelfJobSet, TraceJobSet, DrillJobSet
from tsdb import EvalBusyPeriod
from env_deadline_aware import EnvDeadlineAware
from agent import Agent_AC, Agent_EDF, Agent_FCFS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--mode", choices=['Train', 'Test'], default="Test")  # アルゴリズムを指定
    parser.add_argument("-alg", "--algorithm", choices=['FCFS', 'EDF', 'AC'], default="EDF")  # アルゴリズムを指定
    parser.add_argument('-tr', '--isTraceJobSet', action="store_true")
    parser.add_argument('-dr', '--isDrillJobSet', action="store_true")
    parser.add_argument('-str', '--isStream', action="store_true")
    parser.add_argument("-id", "--inputDir", type=str, default="input")  # 入力用Traceデータファイルのディレクトリ
    parser.add_argument('-i','--inputFileName',default="data_jobset.txt", help='File name of Input JobSet data (default: data_jobset.txt)')  
    parser.add_argument('-pj','--pjName',default='pjName', help='Base of output file name (default: pjName)')  
    parser.add_argument('-bpl','--bplName',default='bplName', help='Base of busy period list file name (default: bplName)')  
    parser.add_argument('-jbs','--jbsName',default='jbsName', help='Base of jobset file name (default: jbsName)')  
    parser.add_argument('-ml','--modelName',default='same', help='Base of output file name (default: same)')  
    #
    parser.add_argument('-lr','--learning_rate',type=float, default=1.0e-5, help='learning rate (default: 1.0e-5)')  # 強化学習の学習率
    parser.add_argument('-g','--gamma',type=float, default=0.99, help='reward discount rate (default: 0.99)')  # 強化学習の報酬の割引率
    parser.add_argument('-ep','--num_episodes',type=int, default=1, help='number of episodes (default: 1)')  
    parser.add_argument('-e','--tmax',type=int, default=500, help='tmax, episode length (default: 500)')  
    parser.add_argument('-s','--seed',type=int, default=0, help='seed (default: 0)')  
    parser.add_argument('-na','--nact',type=int, default=8, help='Nact (default: 8)')  
    parser.add_argument('-m','--mu',type=float, default=1.0, help='mu, decay factor of reward (default: 1.0)')  # 報酬計算の減衰係数 exp(-mu*delay) gamma=1.0 (Sagisaka2022)
    parser.add_argument('-l','--lambda_',type=float, default=0.08, help='lambda, arrival rate per slot (default:0.08)')  # ポアソン過程のパラメータ（1スロットに到着するジョブ数を決める）
    parser.add_argument('-al','--alpha',type=float, default=2.0, help='alpha, defining the deadline by k-times of data size (default: 2.0)')  # 平均デッドライン長
    parser.add_argument('-bt','--beta',type=float, default=10.0, help='beta, defining the job size (default: 10.0')  # 平均ジョブサイズ
    parser.add_argument('-th','--perf_thresh',type=float, default=1.0, help='perf threshold for busy period as a drill (default: 1.0)')  
    parser.add_argument('-nbp','--num_bp',type=int, default=1000000, help='number of busy periods as a drill (default: all)')  
    parser.add_argument('-nr','--nrep',type=int, default=100, help='number of repeat of one pattern before go to the next pattern in drill mode (default: 100)')  
    #

    args = parser.parse_args()
    mode = args.mode
    algorithm = args.algorithm
    print(f"# mode: {mode}, algorithm:{algorithm}")
    isTraceJobSet = args.isTraceJobSet
    isDrillJobSet = args.isDrillJobSet
    isStream = args.isStream
    inputDir = args.inputDir
    inputFileName = args.inputDir+"/"+args.inputFileName
    print(f'# input file\'s name is {inputFileName}.')
    pjName = args.pjName
    bpl_name = args.bplName
    jbs_name = args.jbsName
    if args.modelName == "same":
        modelName = pjName
    else:
        modelName = args.modelName
    num_episodes = args.num_episodes
    tmax = args.tmax
    lr = args.learning_rate
    gamma = args.gamma
    seed = args.seed
    nact = args.nact
    mu = args.mu
    lambda_ = args.lambda_
    alpha = args.alpha
    beta = args.beta
    perf_thresh = args.perf_thresh
    num_bp = args.num_bp
    nrep = args.nrep
    if isDrillJobSet == True:
        num_episodes = num_episodes * nrep
    else:
        num_episodes = num_episodes
    pkl_dir_path = Path('pkl')
    if not pkl_dir_path.exists():
        pkl_dir_path.mkdir()
    pkl_fname = pkl_dir_path.joinpath(f'jobset_{jbs_name}.pkl')
    ofileName_base = "output/odata_"+bpl_name

    # 最初にここでJobSetを作っておく
    if isTraceJobSet == True:
        jobset = TraceJobSet(tmax = tmax,
                             f_name = inputFileName)
    elif isDrillJobSet == True:
        eval_busyperiod = EvalBusyPeriod(ofileName_base = ofileName_base)
        with open(pkl_fname, 'rb') as p:
            loaded_jobset = pickle.load(p)
        print(f"perf_thresh:{perf_thresh}")
        list_bp = eval_busyperiod.from_csv(perf_thresh=perf_thresh,
                                           sample_size = num_bp)
        jobset = DrillJobSet(jobset = loaded_jobset,
                             list_bp = list_bp,
                             nrep = nrep)
        jobset.show_drill_config()
    else:
        jobset = SelfJobSet(tmax = tmax,
                            seed = seed,
                            isStream = isStream,
                            lambda_ = lambda_,
                            alpha = alpha,
                            beta = beta)
        with open(pkl_fname, 'wb') as p:
            pickle.dump(jobset, p)

    #

    env = EnvDeadlineAware(seed = seed,
                           jobset = jobset,
                           nact = nact,
                           mu = mu)

    if algorithm == "FCFS":
        agent = Agent_FCFS(env = env,
                           seed = seed,
                           pjName = pjName,
                           tmax = tmax)
    elif algorithm == "EDF":
        agent = Agent_EDF(env = env,
                          seed = seed,
                          pjName = pjName,
                          tmax = tmax)
    elif algorithm == "AC":
        agent = Agent_AC(env = env,
                         seed = seed,
                         pjName = pjName,
                         tmax = tmax,
                         #
                         modelName = modelName,
                         lr = lr,
                         gamma = gamma)
    else:
        print("Algorithm Not defined.")
        sys.exit(1)


    if mode == "Train":
        agent.train(num_episodes = num_episodes)
    elif mode == "Test":
        agent.test()
        
    
