import numpy as np
import pandas as pd

# Functions to compute velocity and acceleration 
def velocity(position, time):
    """
    Simple v = r/t calc, gives v at each new time step
    Inputs:
        position = [ [x0, y0], [x1, y1], [x2, y2], [x3, y3] ]
        time = [t0, t1, t2, t3]
    """
    positions = np.array(position) # transform data into numpy arrays
    time = np.array(time)
    velocity = np.diff(positions, axis=0) / np.diff(time).reshape(-1, 1) # give arrays correct shape, compute velocity
    return velocity

# OR

def velocity_split(xpos, ypos, time):
    """
    Same v =r/t calc as func(velocity), but accepts separate position inputs for x and y
    e.g. xpos = [x0, x1, x2, x3]
         ypos = [y0, y1, y2, y3]
    """
    positions = np.array([xpos, ypos]).T # transform split data into singular numpy array
    time = np.array(time)
    velocity = np.diff(positions, axis=0) / np.diff(time).reshape(-1, 1) # give arrays correct shape, compute velocity
    return velocity

def acceleration(velocity, time):
    """
    a = v/t calc, gives acceleration at each new time step
    """
    time = np.array(time)
    acceleration = np.diff(velocity, axis=0) / np.diff(time[:-1]).reshape(-1, 1) # give arrays correct shape, compute acceleration
    return acceleration

# QUADRUPOLE CALCS

# Quadrupole moment snapshot
def quadrupole_moment(masses, positions):
    """ 
    Calculate the 2D quadrupole moment for a pair of masses (masses = [m1, m2]). 
    
    Insert data as list of masses and positions (positions = [[x1, y1], [x2, y2]]) 
    """
    Q = np.zeros((2,2)) # 2x2 matrix for quadrupole moment tensor in 2D
    data = zip(masses, positions) # data = [(m1, [x1, y1]), (m2, [x2, y2])]
    for mass, pos in data:
        pos = np.array(pos)
        R = np.dot(pos, pos) # R = [x^2 + y^2 for given m]
        for i in range(2):
            for j in range(2):
                Q[i, j] += mass * (3 * pos[i] * pos[j] - (R if i == j else 0))
    return Q

# Quadrupole moment along a trajectory
def qdrpl_trajectory(masses, trajectories):
    """
    Find the quadrupole moment at each point in the 2D trajectory of two masses.
    Insert data as list of masses and trajectories, with list of trajectories being a list of positions for each mass

    Input:
    masses = [m1, m2]
    trajectories = [
                    [[x1_m1, y1_m1], [x2_m1, y2_m1], [...], ...], 
                    [[x1_m2, y1_m2], [x2_m2, y2_m2], [...], ...]
    ]
    
    Outputs a list of quadrupole moment tensors (2x2) at each position / time step
    For exporting data use list(quadrupole_moments), for dispaying use enumerate(quadrupole moments)
    """
    # Time step N
    time_n = len(trajectories[0])
    
    quadrupole_moments = []
    
    for t in range(time_n):
        # Get positions of all masses at time step t
        pos_t = [trajectory[t] for trajectory in trajectories]
        
        # Calculate the quadrupole moment for this time step
        Q_t = quadrupole_moment(masses, pos_t)
        
        # Append the quadrupole moment tensor to the list
        quadrupole_moments.append(Q_t)
    
    return quadrupole_moments