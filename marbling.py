"""
Computational Marbling - A fluid dynamics simulation for generating marbling art.

This module implements a 2D Navier-Stokes solver for simulating the fluid dynamics
of marbling paper art. It supports both Eulerian (grid-based) and Lagrangian 
(particle-based) descriptions of the fluid.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates


class NavierStokesSolver:
    """
    2D incompressible Navier-Stokes solver using finite differences.
    
    The Navier-Stokes equations describe fluid flow:
    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
    ∇·u = 0 (incompressibility)
    
    Where:
    - u is the velocity field
    - p is pressure
    - ρ is density
    - ν is kinematic viscosity
    - f is external forces
    """
    
    def __init__(self, width=128, height=128, viscosity=0.0001, dt=0.1):
        """
        Initialize the Navier-Stokes solver.
        
        Args:
            width: Grid width
            height: Grid height
            viscosity: Kinematic viscosity (controls how thick/thin the fluid is)
            dt: Time step size
        """
        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.dt = dt
        
        # Eulerian velocity fields (grid-based description)
        self.u = np.zeros((height, width))  # x-velocity
        self.v = np.zeros((height, width))  # y-velocity
        self.u_prev = np.zeros((height, width))
        self.v_prev = np.zeros((height, width))
        
        # Pressure field
        self.p = np.zeros((height, width))
        
    def add_force(self, x, y, fx, fy, radius=5):
        """Add a localized force to the velocity field."""
        x_idx = int(x * self.width)
        y_idx = int(y * self.height)
        
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        dist = np.sqrt((x_grid - x_idx)**2 + (y_grid - y_idx)**2)
        mask = dist <= radius
        
        self.u[mask] += fx
        self.v[mask] += fy
    
    def diffuse(self, field, field_prev, diff_rate):
        """
        Diffusion step using implicit method (stable).
        Solves: field = field_prev + diff_rate * ∇²field
        """
        a = self.dt * diff_rate * self.width * self.height
        
        # Gauss-Seidel relaxation
        for _ in range(20):
            field[1:-1, 1:-1] = (field_prev[1:-1, 1:-1] + a * (
                field[2:, 1:-1] + field[:-2, 1:-1] +
                field[1:-1, 2:] + field[1:-1, :-2]
            )) / (1 + 4 * a)
            
            # Boundary conditions (no-slip)
            field[0, :] = field[1, :]
            field[-1, :] = field[-2, :]
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
    
    def advect(self, field, field_prev, u, v):
        """
        Advection step - move quantities along velocity field.
        This implements the Lagrangian aspect: following particles.
        """
        dt0 = self.dt * max(self.width, self.height)
        
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                # Trace particle backward in time
                x = j - dt0 * u[i, j]
                y = i - dt0 * v[i, j]
                
                # Clamp to boundaries
                x = np.clip(x, 0.5, self.width - 1.5)
                y = np.clip(y, 0.5, self.height - 1.5)
                
                # Bilinear interpolation
                i0, i1 = int(y), int(y) + 1
                j0, j1 = int(x), int(x) + 1
                
                s1 = x - j0
                s0 = 1 - s1
                t1 = y - i0
                t0 = 1 - t1
                
                field[i, j] = (t0 * (s0 * field_prev[i0, j0] + s1 * field_prev[i0, j1]) +
                              t1 * (s0 * field_prev[i1, j0] + s1 * field_prev[i1, j1]))
    
    def project(self):
        """
        Projection step - enforce incompressibility (∇·u = 0).
        This removes the divergent component of velocity.
        """
        h = 1.0 / max(self.width, self.height)
        
        # Compute divergence
        div = np.zeros((self.height, self.width))
        div[1:-1, 1:-1] = -0.5 * h * (
            self.u[1:-1, 2:] - self.u[1:-1, :-2] +
            self.v[2:, 1:-1] - self.v[:-2, 1:-1]
        )
        
        # Solve for pressure using Poisson equation
        self.p = np.zeros((self.height, self.width))
        for _ in range(20):
            self.p[1:-1, 1:-1] = (div[1:-1, 1:-1] + 
                                  self.p[2:, 1:-1] + self.p[:-2, 1:-1] +
                                  self.p[1:-1, 2:] + self.p[1:-1, :-2]) / 4
            
            # Boundary conditions
            self.p[0, :] = self.p[1, :]
            self.p[-1, :] = self.p[-2, :]
            self.p[:, 0] = self.p[:, 1]
            self.p[:, -1] = self.p[:, -2]
        
        # Subtract pressure gradient from velocity
        self.u[1:-1, 1:-1] -= 0.5 * (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / h
        self.v[1:-1, 1:-1] -= 0.5 * (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / h
    
    def step(self):
        """Perform one simulation step."""
        # Save previous state
        self.u_prev[:] = self.u
        self.v_prev[:] = self.v
        
        # Velocity step: diffusion
        self.diffuse(self.u, self.u_prev, self.viscosity)
        self.diffuse(self.v, self.v_prev, self.viscosity)
        
        # Project to ensure incompressibility
        self.project()
        
        # Save state after diffusion and projection
        self.u_prev[:] = self.u
        self.v_prev[:] = self.v
        
        # Velocity step: advection (Lagrangian aspect)
        self.advect(self.u, self.u_prev, self.u_prev, self.v_prev)
        self.advect(self.v, self.v_prev, self.u_prev, self.v_prev)
        
        # Final projection
        self.project()


class DyeParticles:
    """
    Lagrangian particle system for tracking dye/pigment in the fluid.
    Particles follow the fluid velocity field.
    """
    
    def __init__(self, num_particles, color, bounds=(1.0, 1.0)):
        """
        Initialize dye particles.
        
        Args:
            num_particles: Number of particles
            color: RGB color tuple (r, g, b) where each component is in [0, 1]
            bounds: (width, height) of the domain in normalized coordinates
        """
        self.positions = np.random.rand(num_particles, 2) * bounds
        self.color = np.array(color)
        self.num_particles = num_particles
        self.bounds = bounds
    
    def advect(self, u, v, dt):
        """
        Move particles according to velocity field (Lagrangian description).
        
        Args:
            u: x-velocity field
            v: y-velocity field
            dt: time step
        """
        height, width = u.shape
        
        # Scale particle positions to grid coordinates
        grid_x = self.positions[:, 0] * width
        grid_y = self.positions[:, 1] * height
        
        # Clamp to valid range
        grid_x = np.clip(grid_x, 0, width - 1.001)
        grid_y = np.clip(grid_y, 0, height - 1.001)
        
        # Interpolate velocity at particle positions
        coords = np.array([grid_y, grid_x])
        vel_x = map_coordinates(u, coords, order=1, mode='nearest')
        vel_y = map_coordinates(v, coords, order=1, mode='nearest')
        
        # Update positions (Euler integration)
        self.positions[:, 0] += vel_x * dt / width
        self.positions[:, 1] += vel_y * dt / height
        
        # Keep particles in bounds
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.bounds[0])
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.bounds[1])


class CombTool:
    """
    Comb tool for creating marbling patterns by dragging through the fluid.
    """
    
    def __init__(self, num_tines=5, spacing=0.05):
        """
        Initialize comb tool.
        
        Args:
            num_tines: Number of comb tines
            spacing: Distance between tines
        """
        self.num_tines = num_tines
        self.spacing = spacing
        self.current_pos = None
        self.previous_pos = None
    
    def apply(self, solver, start_pos, end_pos, strength=5.0):
        """
        Apply comb motion to the fluid, creating forces along the comb path.
        
        Args:
            solver: NavierStokesSolver instance
            start_pos: (x, y) starting position in normalized coordinates [0, 1]
            end_pos: (x, y) ending position in normalized coordinates [0, 1]
            strength: Force strength multiplier
        """
        # Calculate motion direction
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Perpendicular direction for tine spacing
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return
        
        perp_x = -dy / length
        perp_y = dx / length
        
        # Apply force at each tine
        for i in range(self.num_tines):
            # Offset from center
            offset = (i - (self.num_tines - 1) / 2) * self.spacing
            
            # Tine position
            tine_x = start_pos[0] + perp_x * offset
            tine_y = start_pos[1] + perp_y * offset
            
            # Apply force along motion direction
            if 0 <= tine_x <= 1 and 0 <= tine_y <= 1:
                solver.add_force(tine_x, tine_y, dx * strength, dy * strength, radius=3)


class MarblingSimulation:
    """
    Main simulation class combining fluid solver and particle system.
    """
    
    def __init__(self, width=128, height=128, viscosity=0.0001):
        """
        Initialize marbling simulation.
        
        Args:
            width: Simulation grid width
            height: Simulation grid height
            viscosity: Fluid viscosity
        """
        self.solver = NavierStokesSolver(width, height, viscosity)
        self.dye_layers = []
        self.comb = CombTool()
        self.time = 0
    
    def add_dye_drop(self, x, y, color, num_particles=1000, radius=0.05):
        """
        Add a drop of dye at specified position.
        
        Args:
            x: x-position in normalized coordinates [0, 1]
            y: y-position in normalized coordinates [0, 1]
            color: RGB tuple (r, g, b)
            num_particles: Number of particles in the drop
            radius: Size of the drop
        """
        particles = DyeParticles(num_particles, color, bounds=(1.0, 1.0))
        
        # Position particles in a circular drop
        angles = np.random.rand(num_particles) * 2 * np.pi
        radii = np.random.rand(num_particles) * radius
        particles.positions[:, 0] = x + radii * np.cos(angles)
        particles.positions[:, 1] = y + radii * np.sin(angles)
        
        # Clamp to bounds
        particles.positions[:, 0] = np.clip(particles.positions[:, 0], 0, 1)
        particles.positions[:, 1] = np.clip(particles.positions[:, 1], 0, 1)
        
        self.dye_layers.append(particles)
    
    def apply_comb(self, start_pos, end_pos, strength=5.0):
        """
        Drag a comb through the fluid.
        
        Args:
            start_pos: (x, y) starting position
            end_pos: (x, y) ending position
            strength: Force strength
        """
        self.comb.apply(self.solver, start_pos, end_pos, strength)
    
    def step(self):
        """Advance simulation by one time step."""
        # Update fluid dynamics
        self.solver.step()
        
        # Advect dye particles
        for dye in self.dye_layers:
            dye.advect(self.solver.u, self.solver.v, self.solver.dt)
        
        self.time += self.solver.dt
    
    # Rendering parameters
    PARTICLE_ALPHA = 0.3  # Alpha blending for particle rendering
    
    def render(self, width=800, height=800):
        """
        Render the current state as an image.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            RGB image array
        """
        image = np.ones((height, width, 3))
        
        # Render each dye layer
        for dye in self.dye_layers:
            x_pixels = (dye.positions[:, 0] * width).astype(int)
            y_pixels = (dye.positions[:, 1] * height).astype(int)
            
            # Clamp to image bounds
            x_pixels = np.clip(x_pixels, 0, width - 1)
            y_pixels = np.clip(y_pixels, 0, height - 1)
            
            # Paint particles (vectorized alpha blending for layering effect)
            # Use numpy indexing for better performance
            image[y_pixels, x_pixels] = (
                image[y_pixels, x_pixels] * (1 - self.PARTICLE_ALPHA) + 
                dye.color * self.PARTICLE_ALPHA
            )
        
        return np.clip(image, 0, 1)
    
    def visualize(self, ax=None):
        """
        Visualize the current state.
        
        Args:
            ax: Matplotlib axis to plot on (creates new if None)
            
        Returns:
            The axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        image = self.render()
        ax.imshow(image, origin='lower')
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
        ax.set_aspect('equal')
        ax.axis('off')
        
        return ax


def create_traditional_pattern(steps=100, save_path='marbling_art.png'):
    """
    Create a traditional marbling pattern with multiple colors and comb strokes.
    
    Args:
        steps: Number of simulation steps
        save_path: Path to save the final image
    """
    # Initialize simulation
    sim = MarblingSimulation(width=128, height=128, viscosity=0.00005)
    
    # Add colored dye drops in a pattern
    colors = [
        (0.8, 0.2, 0.2),  # Red
        (0.2, 0.2, 0.8),  # Blue
        (0.9, 0.7, 0.1),  # Yellow
        (0.2, 0.7, 0.3),  # Green
        (0.6, 0.2, 0.7),  # Purple
    ]
    
    # Create a grid of dye drops
    for i, color in enumerate(colors):
        for j in range(3):
            x = 0.2 + (i * 0.15)
            y = 0.3 + (j * 0.2)
            sim.add_dye_drop(x, y, color, num_particles=800, radius=0.06)
    
    # Let dyes spread a bit
    for _ in range(10):
        sim.step()
    
    # Apply comb strokes to create patterns
    # Horizontal strokes
    for y in [0.25, 0.5, 0.75]:
        sim.apply_comb((0.1, y), (0.9, y), strength=8.0)
        for _ in range(5):
            sim.step()
    
    # Vertical strokes
    for x in [0.3, 0.5, 0.7]:
        sim.apply_comb((x, 0.1), (x, 0.9), strength=6.0)
        for _ in range(5):
            sim.step()
    
    # Diagonal stroke
    sim.apply_comb((0.1, 0.1), (0.9, 0.9), strength=4.0)
    for _ in range(5):
        sim.step()
    
    # Continue simulation
    for _ in range(steps):
        sim.step()
    
    # Render and save
    fig, ax = plt.subplots(figsize=(10, 10))
    sim.visualize(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Marbling art saved to {save_path}")
    
    return sim


if __name__ == "__main__":
    # Create a traditional marbling pattern
    sim = create_traditional_pattern(steps=50, save_path='marbling_art.png')
    plt.show()
