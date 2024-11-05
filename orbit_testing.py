import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Load position data from a CSV file
data = pd.read_csv('binary_orbit_data.csv')  # Adjust the path and filename

# Define the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size for better visibility
plt.axis('off')

# Create a larger background
back = plt.Rectangle((-80000, -80000), 160000, 160000, color='#121212')  # Adjust background size

# Create stars with larger sizes
star_1 = plt.Circle((data['Mass1_x'][0], data['Mass1_y'][0]), 2000, color='#3700B3')  # Larger radius
star_2 = plt.Circle((data['Mass2_x'][0], data['Mass2_y'][0]), 2000, color='#BB86FC')  # Larger radius

# Add background and stars to plot
ax.add_patch(back)
ax.add_patch(star_1)
ax.add_patch(star_2)

# Add day label
day_label = ax.text(-70000, -70000, '', fontsize=14, color='cyan')

# Function to update the animation with data from the file
def update(frame):
    # Update Star 1's position using the data
    star_1.center = (data['Mass1_x'][frame], data['Mass1_y'][frame])
    
    # Update Star 2's position using the data
    star_2.center = (data['Mass2_x'][frame], data['Mass2_y'][frame])
    
    # Update the day label
    day_label.set_text(f"Time: {data['Time'][frame]:.1f}")
    
    return star_1, star_2, day_label

# Create the animation
animation = FuncAnimation(fig, update, frames=len(data), interval=100, blit=True)

# Set the axis limits based on your data range
plt.axis('scaled')
plt.xlim(-80000, 80000)  # Adjust limits to match your data
plt.ylim(-80000, 80000)
plt.close()  # Prevents duplicate display of the animation in Colab
plt.rcParams['animation.html'] = 'html5'
HTML(animation.to_jshtml())
