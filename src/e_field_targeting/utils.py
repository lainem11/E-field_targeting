"""
Contains shared, pure mathematical and geometric utility functions.
"""
from typing import Tuple

import jax.numpy as jnp

# --- Type Aliases ---
Array = jnp.ndarray


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
