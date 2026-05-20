import numpy as np


class Perceptron:
    def __init__(self, n_input, n_output, alpha=1.0, theta=0.2):
        self.n_input = n_input
        self.n_output = n_output
        self.alpha = alpha
        self.theta = theta

        self.W = np.zeros((n_input, n_output))
        self.b = np.zeros(n_output)

        self.history = {
            "error_per_epoch": [],
            "weights_per_epoch": [],
            "bias_per_epoch": []
        }

    def activation(self, v):
        if v > self.theta:
            return 1
        elif v < -self.theta:
            return -1
        return 0

    def predict_single(self, x):
        x = np.array(x)
        y = np.zeros(self.n_output)

        for j in range(self.n_output):
            v = np.dot(x, self.W[:, j]) + self.b[j]
            y[j] = self.activation(v)

        return y

    def predict_class(self, x):
        x = np.array(x)

        scores = np.array([
            np.dot(x, self.W[:, j]) + self.b[j]
            for j in range(self.n_output)
        ])

        y = np.array([self.activation(v) for v in scores])
        active = np.where(y == 1)[0]

        if len(active) == 1:
            return active[0]

        return int(np.argmax(scores))

    def confidence_scores(self, x):
        x = np.array(x)

        scores = np.array([
            np.dot(x, self.W[:, j]) + self.b[j]
            for j in range(self.n_output)
        ])

        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    def train(self, X, T, max_epoch=1000, verbose=True):
        X = np.array(X)
        T = np.array(T)

        for epoch in range(1, max_epoch + 1):
            error_count = 0

            for i in range(len(X)):
                x = X[i]
                t = T[i]
                y = self.predict_single(x)

                for j in range(self.n_output):
                    if y[j] != t[j]:
                        error_count += 1
                        self.W[:, j] = self.W[:, j] + self.alpha * t[j] * x
                        self.b[j] = self.b[j] + self.alpha * t[j]

            self.history["error_per_epoch"].append(error_count)
            self.history["weights_per_epoch"].append(self.W.copy())
            self.history["bias_per_epoch"].append(self.b.copy())

            if verbose:
                print(f"Epoch {epoch:4d} | Error: {error_count}")

            if error_count == 0:
                if verbose:
                    print(f"\nTraining konvergen di epoch ke-{epoch}")
                break

        return self.history