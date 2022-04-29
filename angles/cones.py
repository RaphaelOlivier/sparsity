import numpy as np
import torch
from typing import Union
from advertorch.utils import clamp_by_pnorm, _get_norm_batch, batch_multiply
from sambal import random_on_cap
import time

from numpy import cos, dot, empty, log, sin, sqrt, pi
from numpy.random import default_rng
from numpy.linalg import norm

from sambal import random_on_sphere, rotate_from_nth_canonical, random_on_disk


def clamp_by_pnorm_both_ways(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    if (norm == 0).sum() > 0:
        print((norm == 0).sum())
    norm[norm == 0] = 1
    factor = r / norm
    return batch_multiply(factor, x)


def sample_on_unit_sphere(dim, npoints):
    # an array of d normally distributed random variables
    u = np.random.normal(0, 1, (npoints, dim))
    norm = np.sum(u**2, axis=-1) ** (0.5)
    unit_u = u/norm.reshape(-1, 1)
    return unit_u


def angular_projection_batch_directions(batch_points: torch.Tensor, batch_sphere_point: torch.Tensor, cos: float, sin: float):
    batch_size = batch_points.size(0)
    radius_batch = torch.norm(batch_points.view(batch_size, -1), 2, dim=1)
    radius_batch[radius_batch == 0] = 1
    batch_dot_prod = torch.bmm(batch_points.view(
        batch_size, 1, -1), batch_sphere_point.view(batch_size, -1, 1)).view(batch_size)
    cos_angle_point = batch_dot_prod/radius_batch
    proj_axis = (batch_dot_prod.view(batch_size, 1) *
                 batch_sphere_point.view(batch_size, -1)).view(batch_points.size())
    proj_vect = batch_points - proj_axis
    proj_vect_norm = torch.norm(proj_vect.view(batch_size, -1), 2, dim=1)
    proj_vect_norm[proj_vect_norm == 0] = 1
    len_projection = sin * radius_batch / proj_vect_norm * (cos_angle_point < cos) + \
        torch.ones(batch_size, device=batch_points.device) * \
        (cos_angle_point >= cos)
    ndims = [1]*(proj_vect.dim()-1)
    proj_straight = proj_axis + proj_vect*(len_projection.view(-1, *ndims))
    norm_proj_straight = torch.norm(
        proj_straight.view(batch_size, -1), 2, dim=1)
    norm_proj_straight[norm_proj_straight == 0] = 1
    proj_rect = proj_straight * \
        (radius_batch/norm_proj_straight).view(batch_size, *ndims)
    return proj_rect


def get_projection(ord, points, constraint):
    if ord == 2:
        return get_L2_cone_projection(points, constraint)
    else:
        assert ord == np.inf, 'only ord=2 and ord=inf are implemented'
        return get_Linf_projection(points, constraint)


def get_L2_cone_projection(unit_sphere_points, angle):
    angle = torch.tensor(angle)
    cos = torch.cos(angle).to(unit_sphere_points.device)
    sin = torch.sin(angle).to(unit_sphere_points.device)
    #print("new direction batch")

    def project(delta, ord, eps):
        assert ord == 2, "This is L2 cone projection"
        ang_proj = angular_projection_batch_directions(
            delta, unit_sphere_points, cos, sin)
        proj = clamp_by_pnorm_both_ways(ang_proj, ord, eps)
        return proj
    return project


def get_Linf_projection(unit_cube_points, constraint):
    directions = unit_cube_points
    masks = torch.tensor(constraint).to(
        unit_cube_points.device).view(directions.size())

    direction_with_zeros = directions * (1-masks)

    def project(delta, ord, eps):
        assert ord == np.inf, "This is Linf projection"
        diff = delta - eps*direction_with_zeros
        proj = eps*direction_with_zeros + masks*diff
        return proj
    return project


def sample_on_unit_cap(unit_sphere_point, angle, npoints=1):
    points = acc_random_on_cap(unit_sphere_point.reshape(-1), angle,
                               stop_at=npoints).reshape(npoints, *unit_sphere_point.shape)
    return points


def acc_random_on_cap(axis, maximum_planar_angle, rng=default_rng(), stop_at=1, sample_by=1000):
    dim = axis.size
    box_height = (dim-2)*log(sin(min(maximum_planar_angle, pi/2)))
    theta_list = []
    ntheta = 0
    while ntheta < stop_at:
        theta = maximum_planar_angle*rng.random(size=sample_by)
        f = box_height + log(rng.random(size=sample_by))
        inds = np.where(f < (dim-2)*log(sin(theta)))[0]
        if len(inds) > 0:
            theta_list.append(theta[inds])
            ntheta += len(inds)
    theta_list = np.concatenate(theta_list, 0)[:stop_at]
    rotate = [random_on_disk(axis, theta, rng) for theta in theta_list]
    return np.stack(rotate)
