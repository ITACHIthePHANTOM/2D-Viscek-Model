import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000  # Number of nematodes
L = 50  # Size of the box
v0 = 0.1  # Maximum velocity of nematodes
eta = 0.1  # Randomness in direction
r = 2  # Interaction radius

# Initial conditions
np.random.seed(42)
pos = L * np.random.rand(N, 2)  # Random initial positions
theta = 2 * np.pi * np.random.rand(N)  # Random initial directions
vel = v0 * np.column_stack((np.cos(theta), np.sin(theta)))  # Initial velocities

# Simulation
tmax = 501  # Maximum time step
for t in range(tmax):
    # Compute the average direction of neighbors within radius r
    theta_avg = np.zeros(N)
    for i in range(N):
        dist = np.linalg.norm(pos - pos[i], axis=1)
        neighbors = np.where(dist < r)[0]
        theta_avg[i] = np.mean(theta[neighbors])
    
    # Update direction and velocity
    theta += eta * (theta_avg - theta)
    vel = v0 * np.column_stack((np.cos(theta), np.sin(theta)))
    
    # Update position
    pos += vel
    
    # Periodic boundary conditions
    pos[pos < 0] += L
    pos[pos > L] -= L
    
    # Plotting
    plt.clf()
    plt.scatter(pos[:, 0], pos[:, 1], c='b')
    plt.xlim([0, L])
    plt.ylim([0, L])
    plt.title('Time: %d' % t)
    plt.pause(0.001)

plt.show()
