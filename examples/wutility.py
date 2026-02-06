from typing import List, Tuple, Union

import numpy
import trimesh
from trimesh.voxel.creation import voxelize


def get_voxel_pitch(mesh: trimesh.Trimesh, n_cubes: int) -> float:
    """Get the pitch of the voxel grid based on the mesh and number of cubes.

    Args:
        mesh: Mesh to get the pitch from.
        n_cubes: Number of voxels to fit.

    Returns:
        float: Pitch of the voxel grid.
    """
    d = mesh.extents
    cube_volume = d[0] * d[1] * d[2]
    v = mesh.volume
    if v > 0:
        occupancy = 1.0 - ((cube_volume - v) / cube_volume)
    else:
        print("sphere_fit: occupancy test failed, assuming cuboid volume")
        occupancy = 1.0

    # given the extents, find the radius to get number of spheres:
    pitch = (occupancy * cube_volume / n_cubes) ** (1 / 3)
    return pitch


def get_voxelgrid_from_mesh(
    mesh: trimesh.Trimesh, n_spheres: int, voxelize_method: str = "ray"
) -> Tuple[Union[numpy.array, None], Union[numpy.array, None]]:
    """Get voxel grid from mesh using :py:func:`trimesh.voxel.creation.voxelize`.

    Args:
        mesh: Input mesh.
        n_spheres: Number of voxels to fit.
        voxelize_method: Voxelize method. Defaults to "ray".

    Returns:
        Tuple of occupied voxels and side of voxels (length of cube). Returns [None, None] if
            voxelization fails.

    """
    pitch = get_voxel_pitch(mesh, n_spheres)
    radius = pitch / 2.0
    try:
        voxel = voxelize(mesh, pitch, voxelize_method)
        voxel = voxel.fill("base")
        pts = voxel.points
        rad = numpy.ravel([radius for _ in range(len(pts))])
    except:
        print("voxelization failed")
        pts = rad = None
    return pts, rad


def voxel_fit_volume_inside_mesh(
    mesh: trimesh.Trimesh,
    n_spheres: int,
    voxelize_method: str = "ray",
) -> Tuple[numpy.ndarray, numpy.array]:
    """Voxelize mesh, fit spheres to volume. Return the fitted spheres.

    Args:
        mesh: Input mesh.
        n_spheres: Number of spheres to fit.
        voxelize_method: Voxelization method to use, select from
            :py:func:`trimesh.voxel.creation.voxelize`.

    Returns:
        Tuple of sphere positions and their radius.
    """
    pts, rad = get_voxelgrid_from_mesh(mesh, 2 * n_spheres, voxelize_method)
    if pts is None:
        return pts, rad
    # compute signed distance:
    pr = trimesh.proximity.ProximityQuery(mesh)
    dist = pr.signed_distance(pts)

    # calculate distance to boundary:
    dist = dist - rad
    # all negative values are outside the mesh:
    idx = dist > 0.0
    n_pts = pts[idx]
    n_radius = rad[idx].tolist()
    return n_pts, n_radius


def voxel_fit_volume_sample_surface_mesh(
    mesh: trimesh.Trimesh,
    n_spheres: int,
    surface_sphere_radius: float,
    voxelize_method: str = "ray",
) -> Tuple[numpy.ndarray, numpy.array]:
    """Voxelize mesh, fit spheres to volume, and sample surface for points.

    Args:
        mesh: Input mesh.
        n_spheres: Number of spheres to fit.
        surface_sphere_radius: Radius of the spheres on the surface. This radius will be added
            to points on the surface of the mesh, causing the spheres to inflate the mesh volume
            by this amount.
        voxelize_method: Voxelization method to use, select from
            :py:func:`trimesh.voxel.creation.voxelize`.
    Returns:
        Tuple of sphere positions and their radius.
    """
    pts, rad = voxel_fit_volume_inside_mesh(mesh, 0.75 * n_spheres, voxelize_method)
    if pts is None:
        return pts, rad
    # compute surface points:
    if len(pts) >= n_spheres:
        return pts, rad

    sample_count = n_spheres - (len(pts))

    surface_sample_pts, sample_radius = sample_even_fit_mesh(
        mesh, sample_count, surface_sphere_radius
    )
    pts = numpy.concatenate([pts, surface_sample_pts])
    rad = numpy.concatenate([rad, sample_radius])
    return pts, rad


def sample_even_fit_mesh(
    mesh: trimesh.Trimesh,
    n_spheres: int,
    sphere_radius: float,
) -> Tuple[numpy.array, List[float]]:
    """Sample even points on the surface of the mesh and return them with the given radius.

    Args:
        mesh: Mesh to sample points from.
        n_spheres: Number of spheres to sample.
        sphere_radius: Sphere radius.

    Returns:
        Tuple of points and radius.
    """

    n_pts = trimesh.sample.sample_surface_even(mesh, n_spheres)[0]
    n_radius = [sphere_radius for _ in range(len(n_pts))]
    return n_pts, n_radius
