import matplotlib.pyplot as plt

from data import generate_data, Timer
from lrpy import LogisticRegression as LogisticRegressionPy
from lrcpp import LogisticRegression as LogisticRegressionCpp


jobs = []
for size in [500, 5000, 10000]:
    for mean1 in [(-1,0), (1, 1)]:
        for mean2 in [(0,-1), (1, -1)]:
            jobs.append(generate_data(size, mean1, mean2))

def worker(model, args):
    X, Y = args
    model.compute_coefficients(X, Y)

# Profile the python implementation
model = LogisticRegressionPy()
timer = Timer()
for i, job in enumerate(jobs):
    worker(model, job)
py_time = timer.split()

# Profile the C++ implementation
model = LogisticRegressionCpp()
timer = Timer()
for job in jobs:
    worker(model, job)
cpp_time = timer.split()

# Build a nice graph
labels = ["Python", "C++"]
times = [py_time, cpp_time]
index = range(len(labels))
plt.bar(index, times)
plt.xlabel('Implementation language')
plt.ylabel('Runtime')
plt.xticks(index, labels)
plt.title('Comparison of Python and C++ module runtimes')
plt.savefig("profile.png")
