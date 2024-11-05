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

# Create stars figures
star_1 = plt.Circle((data['Mass1_x'][0], data['Mass1_y'][0]), 2000, color='#3700B3')  
star_2 = plt.Circle((data['Mass2_x'][0], data['Mass2_y'][0]), 2000, color='#BB86FC')  

# Add background and stars to plot
ax.add_patch(back)
ax.add_patch(star_1)
ax.add_patch(star_2)

# Function to update the animation with data from the file
def update(frame):
    # Update Star 1 position
    star_1.center = (data['Mass1_x'][frame], data['Mass1_y'][frame])
    
    # Update Star 2 position
    star_2.center = (data['Mass2_x'][frame], data['Mass2_y'][frame])
    
    return star_1, star_2

# Create animation
animation = FuncAnimation(fig, update, frames=len(data), interval=100, blit=True)

# Set the axis limits based on data range
plt.axis('scaled')
plt.xlim(-80000, 80000) 
plt.ylim(-80000, 80000)
plt.close()  # Prevents duplicate display of the animation
plt.rcParams['animation.html'] = 'html5'
HTML(animation.to_jshtml())
