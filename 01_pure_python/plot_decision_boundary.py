import matplotlib.pyplot as plt

from data import generate_data
from lrpy import LogisticRegression


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


lr = LogisticRegression()
a1, a2, b = lr.compute_coefficients(X, Y)

# Equation of line: a1*x + a2*y + b = 0
# Solved for y: y = (1 / a2) (- b - a1 * x)
x1 = 4/3
y1 = (1 / a2) * (-b - a1 * x1)
x2 = -4/3
y2 = (1 / a2) * (-b - a1 * x2)

xlim = plt.xlim()
ylim = plt.ylim()
plt.plot([x1, x2], [y1, y2], color='g', linestyle='-', linewidth=2)
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig("decision_boundary.png")
