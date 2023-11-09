plt.plot(time, qlen_ts, label="Queue Length")
plt.xlabel("time") # x軸のラベル
plt.ylabel("Queue Length") # y軸のラベル
plt.title('Queue Length as a function of time')
plt.legend()
plt.show()

plt.plot(waitingTime, waitingTimeHist, label="Waiting Time Histogram")
plt.xlabel("Waiting Time") # x軸のラベル
plt.ylabel("Histogram") # y軸のラベル
plt.title('Waiting Time Histogram')
plt.legend()
plt.show()

plt.plot(waitingTime, systemTimeHist, label="System Time Histogram")
plt.xlabel("System Time") # x軸のラベル
plt.ylabel("Histogram") # y軸のラベル
plt.title('System Time Histogram')
plt.legend()
plt.show()


def plotInputXiDataGraphs(odir,
                          prefix,
                          t0,
                          t1,
                          key="time"):
    fileName = odir+ "/" +prefix+ ".csv"
    data = pd.read_csv(fileName, header=0)
    data[key] = pd.to_datetime(data[key], format="%Y-%m-%d %H:%M:%S") # 2005-05-04 15:30:00
    data = data[data[key] > t0]
    data = data[data[key] < t1]
    data.index = data[key]
    key_local = "Normalized Sum (total)"
    fig, ax = plt.subplots(1,1, figsize=(9, 4),sharex=True, sharey=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    ax.plot(data.index,data[key_local])
    ax.set_title("Activity factor")
    # ax.legend()
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])
    ax.axes.set_xlabel(key)
    ax.axes.set_ylabel("Activity factor")
    # plt.tight_layout()
    # plt.legend()
    # ghFileName = "pic/"+prefix+".pdf"
    ghFileName = picdir+prefix+".pdf"
    plt.savefig(ghFileName)
    plt.show()
