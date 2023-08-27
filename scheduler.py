def select_job_EDF(observation):
    selected_index = -1
    edf_time = float('inf')
    for index, job in enumerate(observation):
        # # full attributes
        # observation.append([arriving_time, deadline_length, deadline, job_size, job_size_remain]) # full attributes
        arriving_time = job.pop(0)
        deadline_length = job.pop(0)
        deadline = job.pop(0)
        job_size = job.pop(0)
        job_size_remain = job.pop(0)
        if job_size_remain > 0 and deadline <= edf_time: # job_sizeが0より大きいものが有効なジョブ
            edf_time = deadline
            selected_index = index
    return selected_index


def select_job_FCFS(observation):
    if len(observation) == 0:
        return -1
    # #
    # # full attributes
    # observation.append([arriving_time, deadline_length, deadline, job_size, job_size_remain]) # full attributes
    min_arriving_time = float('inf')
    selected_index = -1
    for index, job in enumerate(observation):
        arriving_time = job.pop(0)
        deadline_length = job.pop(0)
        deadline = job.pop(0)
        job_size = job.pop(0)
        job_size_remain = job.pop(0)
        if job_size_remain > 0 and arriving_time <= min_arriving_time: # job_sizeが0より大きいものが有効なジョブ
            min_arriving_time = arriving_time
            selected_index = index
    return selected_index
