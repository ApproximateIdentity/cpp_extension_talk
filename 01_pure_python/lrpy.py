import math

class LogisticRegression:
    def compute_coefficients(self, X, Y):
        # Some fixed constants
        dlt = 0.01
        eps = 0.000001
        max_iter = 100
        # Choose starting point at origin.
        a1 = a2 = b = 0
        for _ in range(max_iter):
            da1, da2, db = self._compute_gradient(X, Y, a1, a2, b)
            if (da1*da1 + da2*da2 + db*db) < eps * eps:
                break
            a1 -= dlt * da1
            a2 -= dlt * da2
            b  -= dlt * db
        return a1, a2, b

    def _compute_gradient(self, X, Y, a1, a2, b):
        # This is the regularization term.
        da1 = a1
        da2 = a2
        db  = b
        # This is the rest of the cost function.
        for (x1, x2), y in zip(X, Y):
            da1 -= h((-y * (a1 * x1 + a2 * x2 + b))) * y * x1
            da2 -= h((-y * (a1 * x1 + a2 * x2 + b))) * y * x2
            db  -= h((-y * (a1 * x1 + a2 * x2 + b))) * y
        return da1, da2, db

# logistic function
def h(z):
    return 1 / (1 + math.exp(-z))
