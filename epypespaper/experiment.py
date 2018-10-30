import time

def repeated_run(f_dict, args, n):

    report = dict()

    for f_key, f in f_dict.items():

        report[f_key] = {'durations': [], 'returns': []}

        for i in range(n):

            t0 = time.perf_counter()
            res = f(*args)
            t1 = time.perf_counter()

            tau = t1 - t0
            
            report[f_key]['durations'].append(tau)
            report[f_key]['returns'].append(res)

    return report
