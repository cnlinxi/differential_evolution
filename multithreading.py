import multiprocessing as mp
import time

def unwrap_self_f(args, **kwargs):
    return Foo.f(*args, **kwargs)
    
#  Get the number of available CPUs (dictates the number of simultaneous calcs possible.)
cpus = mp.cpu_count()
fib = [1,1,2,3,5,8,13,21]

class Foo(object):

    def f(self, x):
        for i in range(10**6):
            if i % 2:
               i += 1
            else:
               i -= 1
        return x*x

    def async(self, processes=cpus):
        pool = mp.Pool(processes)
        result_list = pool.map(unwrap_self_f, zip([self]*len(fib), fib))
        print(result_list)
    
    def sync(self):
        result_list = [self.f(x) for x in fib]
        print(result_list)
        
    def runner(self):
        start = time.time()
        self.async()
        print cpus, time.time() - start
        start = time.time()
        self.sync()
        print '1', time.time() - start
        
f = Foo()
f.runner()
