"""
Contains functions for visualizing optimization results using Plotly.
"""
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .optimizer import StimulusOptimizer, Mesh
from .utils import project_and_flatten

# --- Type Aliases ---
Array = jnp.ndarray
NpArray = np.ndarray


def _e_to_mag(e_field: Array) -> Tuple[Array, Array, Array]:
    """Helper to calculate E-field magnitude."""
    e_mag = jnp.linalg.norm(e_field, axis=1)
    e_max_idx = jnp.argmax(e_mag)
    e_max = e_mag[e_max_idx]
    return e_mag, e_max, e_max_idx


def _get_stimulated_target_3d(
    e_field: Array, mesh: Mesh, target_pos: Array
) -> Tuple[Array, Array]:
    """Calculates the realized 3D stimulation location and direction."""
    dist_from_target = jnp.linalg.norm(mesh.vertices - target_pos, axis=1)
    neighborhood_mask = dist_from_target < 0.03
    plane_normal = jnp.mean(
        mesh.vertex_normals, where=neighborhood_mask[:, None], axis=0
    )
    plane_normal /= jnp.linalg.norm(plane_normal) + 1e-8

    vertices_2d, basis_2d = project_and_flatten(
        mesh.vertices, plane_normal, target_pos
    )

    e_mag, e_max, _ = _e_to_mag(e_field)
    e_mag_n = e_mag / (e_max + 1e-8)
    weights = (e_mag_n**10)[:, jnp.newaxis]
    loc_2d = jnp.sum(weights * vertices_2d, axis=0) / (
        jnp.sum(weights) + 1e-8
    )

    loc_idx = jnp.argmin(jnp.sum((vertices_2d - loc_2d) ** 2, axis=1))
    direction_3d = e_field[loc_idx, :]
    dir_norm = jnp.linalg.norm(direction_3d)
    dir_normalized_3d = jnp.where(
        dir_norm > 1e-8, direction_3d / dir_norm, jnp.zeros_like(direction_3d)
    )
    loc_3d = (basis_2d @ loc_2d) + target_pos

    return loc_3d, dir_normalized_3d


def visualize_solution(
    optimizer: StimulusOptimizer,
    solution_weights: NpArray,
    target_pos: NpArray,
    target_dir: NpArray,
):
    """Creates an interactive 3D plot of the optimization result using Plotly."""
    target_pos_jax = jnp.asarray(target_pos)
    normalized_weights = solution_weights / optimizer.stimulator.max_current_slope
    e_combined = jnp.einsum("c,cvd->vd", normalized_weights, optimizer.efield_set)
    e_mag, _, _ = _e_to_mag(e_combined)

    realized_pos, realized_dir = _get_stimulated_target_3d(
        e_combined, optimizer.mesh, target_pos_jax
    )

    fig = go.Figure(
        go.Mesh3d(
            x=np.array(optimizer.mesh.vertices[:, 0]),
            y=np.array(optimizer.mesh.vertices[:, 1]),
            z=np.array(optimizer.mesh.vertices[:, 2]),
            i=np.array(optimizer.mesh.faces[:, 0]),
            j=np.array(optimizer.mesh.faces[:, 1]),
            k=np.array(optimizer.mesh.faces[:, 2]),
            intensity=np.array(e_mag),
            colorscale="Viridis",
            colorbar_title="E-Field",
            name="Brain Mesh",
        )
    )

    arrow_scale = np.mean(np.ptp(np.array(optimizer.mesh.vertices), axis=0)) * 0.2
    fig.add_trace(go.Cone(x=[target_pos[0]], y=[target_pos[1]], z=[target_pos[2]], u=[target_dir[0]], v=[target_dir[1]], w=[target_dir[2]], sizeref=arrow_scale, showscale=False, colorscale=[[0, "red"], [1, "red"]], anchor="tip", name="Target"))
    fig.add_trace(go.Cone(x=[realized_pos[0]], y=[realized_pos[1]], z=[realized_pos[2]], u=[realized_dir[0]], v=[realized_dir[1]], w=[realized_dir[2]], sizeref=arrow_scale, showscale=False, colorscale=[[0, "cyan"], [1, "cyan"]], anchor="tip", name="Realized"))

    fig.update_layout(title="Optimization Result: E-Field and Direction", legend=dict(x=0.8, y=0.9), scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"), margin=dict(l=0, r=0, b=0, t=40))

    mesh_center = np.mean(optimizer.mesh.vertices, axis=0)
    distance_factor = np.max(np.std(optimizer.mesh.vertices, axis=0)) * 100
    camera_pos = mesh_center + np.array(optimizer.mesh.average_normal) * distance_factor
    camera = {"eye": {"x": camera_pos[0].item(), "y": camera_pos[1].item(), "z": camera_pos[2].item()},"up": {"x": 0, "y": 0, "z": 1},"center": {"x": mesh_center[0].item(), "y": mesh_center[1].item(), "z": mesh_center[2].item()}}
    fig.update_scenes(camera=camera, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
    fig.show()

def visualize_loss_components(optimizer: StimulusOptimizer):
    """Visualizes loss convergence and the relative contributions of its components."""
    loss_data = optimizer.loss_components
    required_keys = ["best_fitness", "focality", "accuracy", "energy"]
    if not loss_data or not all(key in loss_data for key in required_keys):
        print("Loss components are missing or incomplete.")
        return

    generations = np.arange(len(loss_data["best_fitness"]))
    focality, accuracy, energy = loss_data["focality"], loss_data["accuracy"], loss_data["energy"]
    total_components_safe = focality + accuracy + energy + 1e-9
    focality_pct = (focality / total_components_safe) * 100
    accuracy_pct = (accuracy / total_components_safe) * 100
    energy_pct = (energy / total_components_safe) * 100

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Total Loss Convergence", "Relative Contribution of Loss Components"))
    fig.add_trace(go.Scatter(x=generations, y=loss_data["best_fitness"], mode="lines", name="Total Loss", line=dict(color='royalblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=generations, y=focality_pct, mode='lines', stackgroup='one', name='Focality', line=dict(width=0.5, color='#1f77b4')), row=2, col=1)
    fig.add_trace(go.Scatter(x=generations, y=accuracy_pct, mode='lines', stackgroup='one', name='Accuracy Penalty', line=dict(width=0.5, color='#ff7f0e')), row=2, col=1)
    fig.add_trace(go.Scatter(x=generations, y=energy_pct, mode='lines', stackgroup='one', name='Energy Penalty', line=dict(width=0.5, color='#d62728')), row=2, col=1)

    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_yaxes(title_text="Loss Value (log scale)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Contribution (%)", range=[0, 100], row=2, col=1)
    fig.update_layout(height=800, title_text="Optimization Loss Analysis")
    fig.show()
