import matplotlib.pyplot as plt

from data import generate_data_np as generate_data, Timer
from lrcpp import LogisticRegression as LogisticRegressionCpp


jobs = []
for size in [500, 5000, 10000, 20000]:
    for mean1 in [(-1,0), (1, 1)]:
        for mean2 in [(0,-1), (1, -1)]:
            jobs.append(generate_data(size, mean1, mean2))

def worker(model, args):
    X, Y = args
    model.compute_coefficients(X, Y)

# Profile with differing numbers of threads
times = []
for num_threads in range(1, 9):
    model = LogisticRegressionCpp(num_threads=num_threads)
    timer = Timer()
    for i, job in enumerate(jobs):
        worker(model, job)
    times.append((num_threads, timer.split()))

# Build a nice graph
labels = [n for n, _ in times]
times = [t for _, t in times]
index = range(len(labels))
plt.bar(index, times)
plt.xlabel('Number of threads')
plt.ylabel('Runtime')
plt.xticks(index, labels)
plt.title("C++ speed with varying numbers of threads")
plt.savefig("profile.png")
