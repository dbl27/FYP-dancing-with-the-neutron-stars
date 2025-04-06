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
                                masses_t1: List[MassPoint], 
                                masses_t2: List[MassPoint], 
                                masses_t3: List[MassPoint], 
                                dt: float,
                                r_observer: float,
                                theta: float,
                                phi: float) -> Tuple[float, float]:
        """
        Calculate the gravitational-wave strain components h+ and h× using known positions from CSV data.
    
     Parameters:
        -----------
        masses_t1 : List[MassPoint]
            List of mass points at time t1 (previous time step).
        masses_t2 : List[MassPoint]s
            List of mass points at time t2 (current time step).
        masses_t3 : List[MassPoint]
            List of mass points at time t3 (next time step).
        dt : float
            Time step between the data points.
        r_observer : float
            Distance to observer in meters.
        theta : float
            Polar angle of observer in radians.
        phi : float
            Azimuthal angle of observer in radians.
        
        Returns:
        --------
        tuple
            (h_plus, h_cross) strain components.
        """
        # Calculate quadrupole moment tensors at three time steps
        I1 = self.calculate_moment_tensor(masses_t1)
        I2 = self.calculate_moment_tensor(masses_t2)
        I3 = self.calculate_moment_tensor(masses_t3)
    
        # Calculate the numerical second time derivative using finite differences
        I_ddot = (I3 - 2 * I2 + I1) / (dt * dt)
    
        # Project strain components
        prefactor = self.G / (self.c**4 * r_observer)
    
        h_plus = prefactor * (I_ddot[0, 0] - I_ddot[1, 1]) * \
                 (np.cos(theta)**2 * np.cos(2 * phi) + np.sin(2 * theta))
    
        h_cross = 2 * prefactor * I_ddot[0, 1] * \
                  np.cos(theta) * np.sin(2 * phi)
    
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
