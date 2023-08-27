import sys
import argparse
from common.tools import mkdir

from env_deadline_aware import EnvDeadlineAware
from scheduler import select_job_FCFS
from scheduler import select_job_EDF
from tsdb import TsDatabase


def main(algorithm,
         isTraceJobSet,
         isStream,
         ofileName_base,
         f_name,
         num_episodes,
         tmax,
         seed,
         nact,
         gamma,
         lambda_,
         alpha,
         beta):

    env = EnvDeadlineAware(isTraceJobSet = isTraceJobSet,
                           isStream = isStream,
                           f_name = f_name,
                           tmax = tmax,
                           seed = seed,
                           nact = nact,
                           gamma = gamma,
                           lambda_ = lambda_,
                           alpha = alpha,
                           beta = beta)
    #
    ntick = 10
    tick = int(tmax/ntick)
    #
    for episode in range(num_episodes):
        observation,_ = env.reset()
        accum_reward = 0
        #
        logger = TsDatabase(ofileName_base = ofileName_base)
        for i in range(tmax):
            if algorithm == "FCFS":
                idx = select_job_FCFS(observation)
            elif algorithm == "EDF":
                idx = select_job_EDF(observation)
            else:
                print("Algorithm Not defined.")
                sys.exit(1)
            observation, reward, terminated, truncated, info = env.step(action=idx)
            accum_reward += reward
            logger.add([("AccumReward",accum_reward)])
            time, num_jobs, qlen = env.get_status()
            logger.add([("time",time),("num_jobs",num_jobs),("qlen",qlen)])
            #
            if i%tick == 0:
                env.render()
            if terminated:
                print("terminated.")
                break
        #
        logger.plot()
        logger.to_csv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-alg", "--algorithm", choices=['FCFS', 'EDF'], default="EDF")  # アルゴリズムを指定
    parser.add_argument('-tr', '--isTraceJobSet', action="store_true")
    parser.add_argument('-str', '--isStream', action="store_false")
    parser.add_argument("-id", "--inputDir", type=str, default="input")  # 結果出力先ディレクトリ
    parser.add_argument('-i','--inputFileName',default="data_jobset.txt", help='File name of Input JobSet data')  
    parser.add_argument("-od", "--outputDir", type=str, default="output")  # 結果出力先ディレクトリ
    parser.add_argument('-o','--outFileName',default='o_browserUrls.csv', help='File name of Output data')  
    parser.add_argument('-pj','--pjName',default='pjName', help='File name of Output data')  
    #
    parser.add_argument('-ep','--num_episodes',type=int, default=1, help='number of episodes')  
    parser.add_argument('-e','--tmax',type=int, default=500, help='tmax')  
    parser.add_argument('-s','--seed',type=int, default=0, help='seed')  
    parser.add_argument('-na','--nact',type=int, default=8, help='Nact')  
    parser.add_argument('-g','--gamma',type=float, default=0.1, help='gamma')  
    parser.add_argument('-l','--lambda_',type=float, default=0.08, help='lambda')  
    parser.add_argument('-al','--alpha',type=float, default=2.0, help='alpha, defining the deadline by k-times of data size')  
    parser.add_argument('-bt','--beta',type=float, default=10.0, help='beta, defining the job size')  
    #

    args = parser.parse_args()
    algorithm = args.algorithm
    isTraceJobSet = args.isTraceJobSet
    isStream = args.isStream
    print(f"isStream:{isStream}")
    inputDir = args.inputDir
    inputFileName = args.inputDir+"/"+args.inputFileName
    print(f'# input file\'s name is {inputFileName}.')
    outputDir = args.outputDir
    mkdir(outputDir)
    outputFileName = args.outputDir+"/"+args.outFileName
    pjName = args.pjName
    #
    num_episodes = args.num_episodes
    tmax = args.tmax
    # tmax = 12
    seed = args.seed
    nact = args.nact
    gamma = args.gamma
    lambda_ = args.lambda_
    alpha = args.alpha
    beta = args.beta

    main(algorithm = algorithm,
         isTraceJobSet = isTraceJobSet,
         isStream = isStream,
         ofileName_base = outputDir+"/odata_"+pjName,
         f_name = inputFileName,
         num_episodes = num_episodes,
         tmax = tmax,
         seed = seed,
         nact = nact,
         gamma = gamma,
         lambda_ = lambda_,
         alpha = alpha,
         beta = beta)
    
