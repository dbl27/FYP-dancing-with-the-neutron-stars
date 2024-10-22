import numpy as np
from typing import List, Tuple


class MassPoint:
    def __init__(self, mass: float, x: float, y: float, vx: float = 0.0, vy: float = 0.0, ax: float = 0.0, ay: float = 0.0):
        """
        Parameters
        ----------
        mass : float
            DESCRIPTION.
        x : float
            DESCRIPTION.
        y : float
            DESCRIPTION.
        vx : float, optional
            DESCRIPTION. The default is 0.0.
        vy : float, optional
            DESCRIPTION. The default is 0.0.
        ax : float, optional
            DESCRIPTION. The default is 0.0.
        ay : float, optional
            DESCRIPTION. The default is 0.0.

        Returns
        -------
        None.

        """
    
    #properties we want of the each masspoint, we are using 'self' to allow us to define these individually for each mass
        self.mass = float(mass)
        self.position = np.array([float(x), float(y)], dtype=float)
        self.velocity = np.array([float(vx), float(vy)], dtype=float)
        self.acceleration = np.array([float(ax), float(ay)], dtype=float)
    #The @property decorator is defining a method as a property, accessing as an atriubute, but behaving as a method.
    #This will give flexibility should we introduce more dimensions or objects
    @property
    def x(self) -> float:
        return self.position[0]     
    @property
    def y(self) -> float:
        return self.position[1]
    @property
    def xv(self) -> float:
        return self.velocity[0]  
    @property
    def yv(self) -> float:
        return self.velocity[1]
    @property
    def xa(self) -> float:
        return self.acceleration[0]  
    @property
    def ya(self) -> float:
        return self.acceleration[1]

    def updatestate(self, dt: float) -> None:
        #Euler method used for updating
        self.position += self.velocity * dt
        
        # Update velocity using current acceleration
        self.velocity += self.acceleration * dt
    
    def set_acceleration(self, xa: float, ya: float) -> None:
        """update the acceleration arrays"""
        self.acceleration[:] = [xa, ya]
    
    def get_kinetic_energy(self) -> float:
        """Calculate and return kinetic energy"""
        return 0.5 * self.mass * np.sum(self.velocity ** 2)

class GWStrainCalculator:
   #Strain formula
    def __init__(self, G: float = 6.67430e-11, c: float = 299792458.0):
        self.G = G
        self.c = c  
    def calculate_moment_tensor(self, masses: List[MassPoint]) -> np.ndarray:
        """
        Function for mass quadrupole moment tensor
        
        Parameters:
        -----------
        masses : List[MassPoint]
            List of mass points with their current states (as listed)
        Returns:
        --------
        np.ndarray
        3x3 quadrupole moment tensor
        """
        I = np.zeros((3, 3))
        
        for mass in masses:
            r = np.array([mass.x, mass.y, 0.0])
            
            # Construct quadrupole moment tensor
            # I_ij = ∑ m (3x_i x_j - r² δ_ij)
            for i in range(3):
                for j in range(3):
                    if i < 2 and j < 2:  # Only x and y components
                        I[i,j] += mass.mass * (3 * r[i] * r[j])
                        if i == j:
                            I[i,j] -= mass.mass * np.sum(r**2)
        
        return I
    
    def calculate_strain_components(self, 
                                  masses: List[MassPoint], 
                                  dt: float,
                                  r_observer: float,
                                  theta: float,
                                  phi: float) -> Tuple[float, float]:
        """
        Calculate the gravitational-wave strain components h+ and h×
        Parameters:
        -----------
        masses : List[MassPoint]
            List of mass points
        dt : float
            Time step for numerical derivatives
        r_observer : float
            Distance to observer in meters
        theta : float
            Polar angle of observer in radians
        phi : float
            Azimuthal angle of observer in radians
        Returns:
        --------
        tuple
            (h_plus, h_cross) strain components
        """
        # Store current positions
        original_positions = [(m.x, m.y) for m in masses]
        
        # Calculate first quadrupole moment
        I1 = self.calculate_moment_tensor(masses)
        
        # Update positions using velocities
        for mass in masses:
            mass.updatestate(dt)
            
        # Calculate second quadrupole moment
        I2 = self.calculate_moment_tensor(masses)
        
        # Calculate numerical second time derivative
        I_ddot = (I2 - I1) / (dt * dt)
        
        # Reset positions
        for mass, (x, y) in zip(masses, original_positions):
            mass.position[:] = [x, y]
        
        # Project strain components
        prefactor = self.G / (self.c**4 * r_observer)
        
        h_plus = prefactor * (I_ddot[0,0] - I_ddot[1,1]) * \
                 (np.cos(theta)**2 * np.cos(2*phi) + np.sin(2*theta))
        
        h_cross = 2 * prefactor * I_ddot[0,1] * \
                  np.cos(theta) * np.sin(2*phi)
        
        return h_plus, h_cross
    
    def calculate_strain_amplitude(self, h_plus: float, h_cross: float) -> float:
        """
        Calculate the total strain amplitude
        
        Parameters:
        -----------
        h_plus : float
            Plus polarization component
        h_cross : float
            Cross polarization component
            
        Returns:
        --------
        float
            Total strain amplitude
        """
        return np.sqrt(h_plus**2 + h_cross**2)

def simulate_binary_system(mass1: float, mass2: float, 
                         separation: float, orbital_period: float,
                         num_steps: int) -> List[List[MassPoint]]:
    """
    Create a simple circular binary system for testing
    
    Parameters:
    -----------
    mass1, mass2 : float
        Masses in kg
    separation : float
        Initial separation in meters
    orbital_period : float
        Orbital period in seconds
    num_steps : int
        Number of time steps to simulate
        
    Returns:
    --------
    List[List[MassPoint]]
        List of mass point states at each time step
    """
    dt = orbital_period / num_steps
    omega = 2 * np.pi / orbital_period
    
    trajectory = []
    
    for t in range(num_steps):
        angle = omega * t * dt
        
        # Calculate positions for a circular orbit
        x1 = (mass2 / (mass1 + mass2)) * separation * np.cos(angle)
        y1 = (mass2 / (mass1 + mass2)) * separation * np.sin(angle)
        
        x2 = -(mass1 / (mass1 + mass2)) * separation * np.cos(angle)
        y2 = -(mass1 / (mass1 + mass2)) * separation * np.sin(angle)
        
        # Calculate velocities
        vx1 = -omega * y1
        vy1 = omega * x1
        vx2 = -omega * y2
        vy2 = omega * x2
        
        m1 = MassPoint(mass1, x1, y1, vx1, vy1)
        m2 = MassPoint(mass2, x2, y2, vx2, vy2)
        
        trajectory.append([m1, m2])
    
    return trajectory
"///////////////////////////////////////////////////////////////////////////////////////"
mass1 = 1.4 * 2e30
mass2 = 1.4 * 2e30
separation = 1e5
period = 0.01
num_steps = 1000

# Generate trajectory
trajectory = simulate_binary_system(mass1, mass2, separation, period, num_steps)

# Calculate strain
calculator = GWStrainCalculator()
r_observer = 1e20  # 10 kpc
theta = 0          # overhead observer
phi = 0

# Calculate strain for each timestep
strains = []
dt = period / num_steps

for masses in trajectory:
    h_plus, h_cross = calculator.calculate_strain_components(
        masses, dt, r_observer, theta, phi
    )
    strain_amplitude = calculator.calculate_strain_amplitude(h_plus, h_cross)
    strains.append(strain_amplitude)

# Plot results
import matplotlib.pyplot as plt
time = np.linspace(0, period, num_steps)
plt.figure(figsize=(10, 6))
plt.plot(time, strains)
plt.xlabel('Time (s)')
plt.ylabel('Strain Amplitude')
plt.title('Gravitational Wave Strain from Binary Neutron Star System')
plt.grid(True)
plt.show()









