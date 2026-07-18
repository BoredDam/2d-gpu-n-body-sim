import csv
import numpy as np
import matplotlib.pyplot as plt

SEED = 42


def make_galaxy(
    N,
    radius,
    rad_noise,
    body_speed,
    center,
    seed
):
    np.random.seed(SEED)
    X = []
    Y = []
    vX = []
    vY = []
    mass = []

    for theta in np.linspace(0, 2*np.pi, N - 1):
        X.append(np.cos(theta) * (radius + (np.random.random() - 0.5) * rad_noise) + center[0])
        Y.append(np.sin(theta) * (radius + (np.random.random() - 0.5) * rad_noise) + center[1])
        vX.append(np.cos(theta + 3/2 * np.pi) * body_speed)
        vY.append(np.sin(theta + 3/2 * np.pi) * body_speed)
        mass.append(1)

    X.append(center[0])
    Y.append(center[1])
    vX.append(0)
    vY.append(0)
    mass.append(10000)
    return X, Y, vX, vY, mass

def plot_galaxy(X, Y, vX, vY, mass):
    plt.scatter(X, Y, s=mass)
    plt.quiver(X, Y, vX, vY, alpha=0.2)
    return plt.show()

def write_galaxy(name, X, Y, vX, vY, mass, round_to=7):
    with open(f'./galaxies/{name}.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(X)):
            writer.writerow([
                np.round(X[i], round_to), 
                np.round(Y[i], round_to),
                np.round(vX[i], round_to),
                np.round(vY[i], round_to), 
                mass[i]]
            )
    return True

X, Y, vX, vY, mass = make_galaxy(
    100,
    20,
    6,
    14,
    (-70, 0),
    SEED
)

X1, Y1, vX1, vY1, mass1 = make_galaxy(
    100,
    12,
    6,
    -10,
    (20, 0),
    SEED
)


X, Y, vX, vY, mass = X + X1, Y + Y1, vX + vX1, vY + vY1, mass + mass1
write_galaxy('test_galaxy5', X, Y, vX, vY, mass)
plot_galaxy(X, Y, vX, vY, mass)
