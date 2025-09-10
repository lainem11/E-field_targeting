"""
Handles loading and preprocessing of stimulator and E-field model data.
"""
import json
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

# --- Import from local package modules ---
from .optimizer import Mesh, Stimulator
from .utils import calculate_vertex_normals

# --- Type Aliases ---
Array = jnp.ndarray


def load_stimulator(filename: str) -> Stimulator:
    """Loads stimulator properties from a JSON file."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        max_current_slope = jnp.atleast_1d(
            jnp.asarray(
                data.get(
                    "max_current_slope",
                    np.array(data["max_voltage"]) / np.array(data["inductance"]),
                ),
                dtype=jnp.float32,
            )
        )
        polarity = jnp.atleast_1d(
            jnp.asarray(
                data.get("polarity", jnp.ones_like(max_current_slope)),
                dtype=jnp.float32,
            )
        )
        stimulator = Stimulator(max_current_slope=max_current_slope, polarity=polarity)

        return stimulator
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        raise IOError(f"Error reading or processing stimulator file: {e}")


def load_efield_model(
    filename: str, stimulator: Stimulator, jax_device
) -> Tuple[Mesh, Array, Array]:
    """Loads E-field and mesh data, preprocesses it, and returns data structures."""
    try:
        efile = loadmat(filename)
        efield_set = jnp.asarray(
            np.stack(efile["Efield"].flatten()), dtype=jnp.float32
        )
    except (FileNotFoundError, KeyError) as e:
        raise IOError(f"Failed to load or parse .mat file: {e}")

    max_efield_set = efield_set * stimulator.max_current_slope.reshape(-1, 1, 1)
    efield_set_gpu = jax.device_put(max_efield_set, jax_device)

    mesh_data = {
        "vertices": jnp.asarray(efile["smesh"]["p"][0][0], dtype=jnp.float32),
        "faces": jnp.asarray(
            efile["smesh"]["e"][0][0].astype(np.int32) - 1, dtype=jnp.int32
        ),
    }
    mesh_gpu = {k: jax.device_put(v, jax_device) for k, v in mesh_data.items()}

    vertex_normals = calculate_vertex_normals(mesh_gpu["vertices"], mesh_gpu["faces"])
    mean_normal = jnp.nanmean(vertex_normals, axis=0)
    norm_mean_normal = jnp.linalg.norm(mean_normal)
    avg_normal = jnp.where(
        norm_mean_normal > 1e-8,
        mean_normal / norm_mean_normal,
        jnp.zeros_like(mean_normal),
    )

    mesh = Mesh(
        vertices=mesh_gpu["vertices"],
        faces=mesh_gpu["faces"],
        vertex_normals=vertex_normals,
        average_normal=avg_normal,
    )

    efield_basis = _compute_efield_basis(efield_set_gpu, mesh.average_normal)

    return mesh, efield_set_gpu, efield_basis


def _compute_efield_basis(efield_set: Array, average_normal: Array) -> Array:
    """Computes a 2D basis for the E-field at each vertex using SVD."""
    e_reshaped = efield_set.transpose(1, 0, 2)
    means = e_reshaped.mean(axis=1)
    e_centered = e_reshaped - means[:, None, :]

    _, _, vh = jax.vmap(jnp.linalg.svd)(e_centered)
    v_basis = vh.transpose(0, 2, 1)
    v_top2 = v_basis[:, :, :2]

    plane_normals = jnp.cross(v_top2[..., 0], v_top2[..., 1], axis=-1)
    dot_products = jnp.sum(plane_normals * average_normal, axis=-1)
    flip_mask = dot_products < 0
    flipper = jnp.array([1.0, -1.0])
    efield_basis = v_top2 * jnp.where(
        flip_mask[:, None, None], flipper[None, None, :], 1.0
    )
    return efield_basis
