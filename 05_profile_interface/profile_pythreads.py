from multiprocessing.pool import ThreadPool
from data import Timer
from module import noop


payload = [float(i) for i in range(1000000)]
def worker(arg):
    noop(payload)

jobs = range(200)

print("Threads", "Runtime", sep="\t")
for num_threads in range(1, 9):
    pool = ThreadPool(num_threads)
    timer = Timer()
    pool.map(worker, jobs)
    elapsed_time = timer.split()
    print(num_threads, elapsed_time, sep="\t")
