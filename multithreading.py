import multiprocessing as mp
import time

def foo_pool(x):
    for i in range(10**6):
        if i % 2:
           i += 1
        else:
           i -= 1
    return x*x
    
#  Get the number of available CPUs (dictates the number of simultaneous calcs possible.)
cpus = mp.cpu_count()
fib = [1,1,2,3,5,8,13]


def async(processes=cpus):
    pool = mp.Pool(processes)
    result_list = pool.map(foo_pool, fib)
    print(result_list)
    
def sync():
    result_list = [foo_pool(x) for x in fib]
    print(result_list)

start = time.time()
async()
print cpus, time.time() - start

start = time.time()
async(int(cpus * 0.5))
print int(cpus * 0.5), time.time() - start

start = time.time()
sync()
print '1', time.time() - start
