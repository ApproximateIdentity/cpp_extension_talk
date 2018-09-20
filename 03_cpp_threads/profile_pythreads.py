from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt

from data import generate_data, Timer
from lrcpp import LogisticRegression as LogisticRegressionCpp


jobs = []
for size in [500, 5000, 10000, 20000]:
    for mean1 in [(-1,0), (1, 1)]:
        for mean2 in [(0,-1), (1, -1)]:
            jobs.append(generate_data(size, mean1, mean2))


# Profile with differing numbers of python threads with a model that is
# single-threaded.
model = LogisticRegressionCpp()
def worker(args):
    X, Y = args
    model.compute_coefficients(X, Y)

times = []
for num_threads in range(1, 9):
    pool = ThreadPool(num_threads)
    timer = Timer()
    pool.map(worker, jobs)
    times.append((num_threads, timer.split()))


# Build a nice graph
labels = [n for n, _ in times]
times = [t for _, t in times]
index = range(len(labels))
plt.bar(index, times)
plt.xlabel('Number of threads')
plt.ylabel('Runtime')
plt.xticks(index, labels)
plt.title("C++ speed with varying numbers of python(!) threads")
plt.savefig("profile_pythreads.png")
