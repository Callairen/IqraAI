import numpy as np

X = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
])

T = np.array([1, -1, -1, -1])

alpha = 1
theta = 0.2

w = np.zeros(2)
b = 0

epoch = 0
max_epoch = 100

while True:
    error_count = 0
    epoch += 1

    for i in range(len(X)):
        x = X[i]
        t = T[i]

        v = np.dot(x, w) + b

        if v > theta:
            y = 1
        elif v < -theta:
            y = -1
        else:
            y = 0

        if y != t:
            error_count += 1
            w = w + alpha * t * x
            b = b + alpha * t

    print(f"Epoch {epoch}: error={error_count}, w={w}, b={b}")

    if error_count == 0:
        print(f"\nTraining selesai di epoch ke-{epoch}")
        break

    if epoch >= max_epoch:
        print("Max epoch tercapai")
        break