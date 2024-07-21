import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
num_points = 100  # Number of points along the rope
num_steps = 500   # Number of simulation steps
dt = 0.33        # Time step
a = 0.5          # Parameter of the parabolic function

# Initial conditions
x = np.linspace(-1, 1, num_points)
y = np.sin(x)

# Function to update the rope shape
def update_rope(y):
    # Random perturbation for wiggling effect
    noise = np.random.normal(scale=0.01, size=num_points)
    y += noise
    
    # Clear the plot and plot the updated rope
    plt.clf()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Wiggling Rope Simulation')

# Simulation loop
for step in range(num_steps):
    update_rope(y)
    plt.pause(0.01)

# Keep the plot window open
plt.show()
