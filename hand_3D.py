import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

with open("X.txt", mode="r") as fx:
    X = [eval(line.strip()) for line in fx]

with open("Y.txt", mode="r") as fy:
    y = [int(line.strip()) for line in fy]

num_list = []

for yi in range(len(y)):
    if y[yi] == 1:
        num_list.append(yi)

pre_X = np.array(X)
X = np.zeros_like(pre_X)

for i in range(len(X)):
    pre_X[i, :, :] = pre_X[i, :, :] - pre_X[i, 0, :]

nn = 0
for n in num_list:
    X[nn, :, :] = pre_X[n, :, :]
    nn += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def update_graph(num):
    ax.clear()
    data = X[num]
    x = [point[0] for point in data]
    y = [point[1] for point in data]
    z = [point[2] for point in data]
    for j in range(5):
        ax.plot([x[0], x[4 * j + 1]], [y[0], y[4 * j + 1]], [z[0], z[4 * j + 1]], color="r")
        for k in range(1, 4):
            ax.plot([x[4 * j + k], x[4 * j + k + 1]], [y[4 * j + k], y[4 * j + k + 1]],
                    [z[4 * j + k], z[4 * j + k + 1]], color="b")
    ax.plot([x[2], x[5]], [y[2], y[5]], [z[2], z[5]], color="g")
    for h in range(1, 4):
        ax.plot([x[4 * h + 1], x[4 * (h + 1) + 1]], [y[4 * h + 1], y[4 * (h + 1) + 1]], [z[4 * h + 1], z[4 * (h + 1) + 1]], color="g")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-0.1, 0.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {num + 1}')


ani = FuncAnimation(fig, update_graph, frames=len(X), interval=1000)
plt.show()