import matplotlib.pyplot as plt
from data import generate_data

X, Y = generate_data(50, (-1/3, -1/3), (1/3, 1/3))

reds = [[], []]
blues = [[], []]
for x, y in zip(X, Y):
    if y == -1:
        reds[0].append(x[0])
        reds[1].append(x[1])
    else:
        blues[0].append(x[0])
        blues[1].append(x[1])

plt.plot(reds[0], reds[1], 'ro')
plt.plot(blues[0], blues[1], 'bo')
plt.savefig("points.png")
