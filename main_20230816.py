import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
import argparse

from env_deadline_aware import EnvDeadlineAware
from agent import Agent_AC


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
    parser.add_argument('-th','--perf_thresh',type=float, default=1.0, help='perf threshold for busy period as a drill (default: 1.0')  
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
    pbl_name = args.bplName
    jbs_name = args.jbsName
    if args.modelName == "same":
        modelName = pjName
    else:
        modelName = args.modelName
    #
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

    env = EnvDeadlineAware(isStream = isStream,
                           isTraceJobSet = isTraceJobSet,
                           isDrillJobSet = isDrillJobSet,
                           f_name = inputFileName,
                           tmax = tmax,
                           seed = seed,
                           nact = nact,
                           mu = mu,
                           lambda_ = lambda_,
                           alpha = alpha,
                           beta = beta,
                           perf_thresh = perf_thresh,
                           bplist_name = pbl_name,
                           jobset_name = jbs_name)
    agent = Agent_AC(env = env,
                     seed = seed,
                     pjName = pjName,
                     tmax = tmax,
                     #
                     algorithm = algorithm,
                     modelName = modelName,
                     lr = lr,
                     gamma = gamma)
    if mode == "Train":
        agent.train(num_episodes = num_episodes)
    elif mode == "Test":
        agent.test()
        
    
