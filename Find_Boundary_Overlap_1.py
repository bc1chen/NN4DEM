import torch

"""             2D DOMAIN
        (ly,0)------------(lx,ly) 
           .                 .
           .                 .
           .                 .
        (0,0)--------------(lx,0)

"""

def find_boundary_overlap(x_grid, y_grid, lx, ly, kn):

    # Calculate overlap indicators for each boundary
    is_left_overlap = torch.lt(x_grid, 0)  # Overlap with left wall
    is_right_overlap = torch.gt(x_grid, lx)  # Overlap with right wall
    is_bottom_overlap = torch.lt(y_grid, 0)  # Overlap with bottom wall
    is_top_overlap = torch.gt(y_grid, ly)  # Overlap with top wall

    # Combine overlap indicators across all boundaries
    is_overlapping = torch.any(torch.stack([is_left_overlap, is_right_overlap, is_bottom_overlap, is_top_overlap], dim=1))

    # Calculate penetration depth for each direction
    penetration_left = torch.clamp_min(x_grid, 0)  # Penetration into left wall
    penetration_right = torch.clamp_max(x_grid - lx, 0)  # Penetration into right wall
    penetration_bottom = torch.clamp_min(y_grid, 0)  # Penetration into bottom wall
    penetration_top = torch.clamp_max(y_grid - ly, 0)  # Penetration into top wall

    # Calculate contact force components (positive for inward force)
    fx_grid = - kn * torch.where(is_overlapping & is_left_overlap, penetration_left, torch.zeros_like(penetration_left))
    fx_grid += kn * torch.where(is_overlapping & is_right_overlap, penetration_right, torch.zeros_like(penetration_right))
    fy_grid = - kn * torch.where(is_overlapping & is_bottom_overlap, penetration_bottom, torch.zeros_like(penetration_bottom))
    fy_grid += kn * torch.where(is_overlapping & is_top_overlap, penetration_top, torch.zeros_like(penetration_top))

    return is_overlapping, fx_grid, fy_grid
