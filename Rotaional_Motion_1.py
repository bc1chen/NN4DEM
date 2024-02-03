import torch

def update_positions_and_rotations(x_grid, y_grid, vx_grid, vy_grid, fx_grid, fy_grid, dt, particle_mass, particle_radius, mu, orientations):
    """
    Updates particle positions and rotations, considering both translational and rotational motion with friction.

    Args:
        particle_inertia: Rotational inertia of each particle.
        mu: Friction coefficient.
        
    """
    # Calculate Rotaional Inertia of Each Particle
    # Sphere: I = (2/5) * m * r**2.
    # Disc: I = (1/2) * m * r^2.
    particle_inertia = 0.5*particle_mass*particle_radius**2

    # Calculate tangential forces due to friction
    tangential_x = -mu * fy_grid
    tangential_y = mu * fx_grid

    # Calculate total forces considering friction
    total_fx_grid = fx_grid + tangential_x
    total_fy_grid = fy_grid + tangential_y

    # Calculate torques from tangential forces
    rx_grid, ry_grid = torch.meshgrid(torch.arange(x_grid.shape[1]), torch.arange(x_grid.shape[0]))  # Grid coordinates for torque calculation
    torques = (ry_grid - y_grid) * tangential_x - (rx_grid - x_grid) * tangential_y

    # Calculate torques (assuming frictionless contacts)
    # rx_grid, ry_grid = torch.meshgrid(torch.arange(x_grid.shape[1]), torch.arange(x_grid.shape[0]))  # Grid coordinates for torque calculation
    # torques = (ry_grid - y_grid) * fx_grid - (rx_grid - x_grid) * fy_grid  # Cross product to get torques

    # Update angular velocities
    angular_velocities = particle_inertia * torques / dt

    # Update orientations (using Euler angles for simplicity)
    orientations_new = orientations + angular_velocities * dt

    # Update positions
    x_grid_new = x_grid + vx_grid * dt
    y_grid_new = y_grid + vy_grid * dt

    return x_grid_new, y_grid_new, orientations_new, total_fx_grid, total_fy_grid, angular_velocities
