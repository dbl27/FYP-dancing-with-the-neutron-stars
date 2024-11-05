import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

data = pd.read_csv('binary_orbit_data.csv')

fig, ax = plt.subplots(figsize=(10, 10)) 
plt.axis('off')

# Create background
back = plt.Rectangle((-80000, -80000), 160000, 160000, color='#121212') 

# Create stars
star_1 = plt.Circle((data['Mass1_x'][0], data['Mass1_y'][0]), 2000, color='#3700B3')  
star_2 = plt.Circle((data['Mass2_x'][0], data['Mass2_y'][0]), 2000, color='#BB86FC')  

# Create trajectory lines 
trajectory1, = ax.plot([], [], linestyle='dotted', color='#3700B3', linewidth=1)
trajectory2, = ax.plot([], [], linestyle='dotted', color='#BB86FC', linewidth=1)

# Add background and stars to plot
ax.add_patch(back)
ax.add_patch(star_1)
ax.add_patch(star_2)

# Initialize trajectories
trajectory1.set_data([], [])
trajectory2.set_data([], [])

# Function to update the animation with file data
def update(frame):
    # Update Star 1 position
    star_1.center = (data['Mass1_x'][frame], data['Mass1_y'][frame])
    
    # Update Star 2 position
    star_2.center = (data['Mass2_x'][frame], data['Mass2_y'][frame])
    
    # Update the trajectories
    x1_data = data['Mass1_x'][:frame+1]  # Positions of star 1 up to the current frame
    y1_data = data['Mass1_y'][:frame+1]
    trajectory1.set_data(x1_data, y1_data)

    x2_data = data['Mass2_x'][:frame+1]  # Positions of star 2 up to the current frame
    y2_data = data['Mass2_y'][:frame+1]
    trajectory2.set_data(x2_data, y2_data)

    return star_1, star_2, trajectory1, trajectory2

# Create animation
animation = FuncAnimation(fig, update, frames=len(data), interval=100, blit=True)

# Set the axis limits based on data range
plt.axis('scaled')
plt.xlim(-80000, 80000) 
plt.ylim(-80000, 80000)
plt.close()  # Prevents duplicate display of the animation
plt.rcParams['animation.html'] = 'html5'
HTML(animation.to_jshtml())
