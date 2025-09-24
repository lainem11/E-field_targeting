"""
Contains shared, pure mathematical and geometric utility functions.
"""
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay

# --- Type Aliases ---
Array = jnp.ndarray

def triangulate_vertices(vertices: Array) -> Array:
    """
    Computes a 2D Delaunay triangulation for vertices projected on the XY plane.

    This is suitable for meshes that are structured like a cap or sheet,
    where a 2D projection is a reasonable representation for connectivity.

    Args:
        vertices: A JAX array of shape (n_vertices, 3) representing vertex coordinates.

    Returns:
        A JAX array of shape (n_faces, 3) with integer type, representing
        the vertex indices for each triangular face.
    """
    # Convert to NumPy array for compatibility with SciPy.
    points_2d = np.asarray(vertices[:, :2])

    # Perform Delaunay triangulation on the 2D points.
    tri = Delaunay(points_2d)

    # The 'simplices' attribute contains the indices of the vertices for each triangle.
    # Convert the result back to a JAX array with the appropriate integer type.
    return jnp.asarray(tri.simplices, dtype=jnp.int32)

def project_and_flatten(
    vertices: Array, normal: Array, origin: Array
) -> Tuple[Array, Array]:
    """Projects 3D vertices onto a plane and flattens them to 2D."""
    v_shifted = vertices - origin
    dot_products = v_shifted @ normal
    p_shifted = v_shifted - (dot_products[:, None]) * normal

    basis_vectors = jnp.eye(3)
    cross_products = jnp.cross(normal, basis_vectors)
    max_idx = jnp.argmax(jnp.linalg.norm(cross_products, axis=1))
    u_vec = cross_products[max_idx]
    u_vec /= jnp.linalg.norm(u_vec)

    w_vec = jnp.cross(normal, u_vec)
    w_vec /= jnp.linalg.norm(w_vec)

    basis = jnp.stack([u_vec, w_vec], axis=1)
    coordinates_2d = p_shifted @ basis
    return coordinates_2d, basis


def calculate_vertex_normals(vertices: Array, faces: Array) -> Array:
    """Calculates vertex normals for a triangulated 3D mesh using JAX."""
    v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_normals = jnp.cross(v1, v2)

    face_norms = jnp.linalg.norm(face_normals, axis=1, keepdims=True)
    safe_face_normals = jnp.where(
        face_norms > 1e-8, face_normals / face_norms, jnp.zeros_like(face_normals)
    )

    vertex_normals = jnp.zeros_like(vertices)
    vertex_normals = vertex_normals.at[faces[:, 0]].add(safe_face_normals)
    vertex_normals = vertex_normals.at[faces[:, 1]].add(safe_face_normals)
    vertex_normals = vertex_normals.at[faces[:, 2]].add(safe_face_normals)

    norms = jnp.linalg.norm(vertex_normals, axis=1, keepdims=True)
    safe_vertex_normals = jnp.where(
        norms > 1e-8, vertex_normals / norms, jnp.zeros_like(vertex_normals)
    )

    is_zero_norm = (norms < 1e-8).squeeze()
    default_normal = jnp.array([0.0, 0.0, 1.0])
    return jnp.where(is_zero_norm[:, None], default_normal, safe_vertex_normals)

def generate_target_from_shift(
    target_shift: Array, vertices: Array
) -> Tuple[Array, Array]:
    """
    Generates target position and direction from shift parameters.

    The function assumes a spherical mesh and calculates the target position
    by rotating an apex point on the sphere. The direction is similarly
    calculated by rotating a default direction vector.

    Args:
        target_shift: A JAX array of shape (3,) containing the target shifts:
                      - target_shift[0]: Horizontal shift 'x' in mm.
                      - target_shift[1]: Vertical shift 'y' in mm.
                      - target_shift[2]: Clockwise rotation 'theta' in degrees.
        vertices: A JAX array of shape (n_vertices, 3) representing vertex
                  coordinates on a spherical surface.

    Returns:
        A tuple containing:
        - target_position (Array): A JAX array of shape (3,) for the target coordinates.
        - target_direction (Array): A JAX array of shape (3,) for the target direction vector.
    """
    # Ensure target_shift is a JAX array
    target_shift = jnp.asarray(target_shift, dtype=jnp.float32)

    # Calculate the radius of the sphere from the vertices
    r = jnp.max(jnp.sqrt(jnp.sum(vertices**2, axis=1)))

    # The apex of the mesh, starting point for rotation
    mesh_apex = jnp.array([0.0, 0.0, r])

    # Convert shifts (mm) and rotation (degrees) to angles (radians)
    # phi: rotation around x-axis from vertical shift 'y'
    phi = target_shift[1] / 1000.0 / r
    # pha: rotation around y-axis from horizontal shift 'x'
    pha = -target_shift[0] / 1000.0 / r
    # pho: rotation around z-axis from 'theta'
    pho = jnp.deg2rad(target_shift[2])

    # Define rotation matrices
    def Rmatx(a: float) -> Array:
        s, c = jnp.sin(a), jnp.cos(a)
        return jnp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def Rmaty(a: float) -> Array:
        s, c = jnp.sin(a), jnp.cos(a)
        return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def Rmatz(a: float) -> Array:
        s, c = jnp.sin(a), jnp.cos(a)
        return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    # Calculate target position by rotating the apex
    target_position = mesh_apex @ Rmatx(phi) @ Rmaty(pha)

    # Define the initial, unrotated direction
    center_dir = jnp.array([0.0, 1.0, 0.0])

    # Calculate target direction by rotating the center direction
    target_direction = center_dir @ Rmatz(pho) @ Rmatx(phi) @ Rmaty(pha)

    return target_position, target_direction