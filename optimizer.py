"""
Performs stimulus optimization for transcranial magnetic stimulation (TMS)
using Differential Evolution.

This module defines the `StimulusOptimizer` class, which encapsulates the logic
for loading E-field models, setting up, and running an optimization process to
find the optimal coil currents for a desired stimulation target on a cortical
surface mesh.

The optimization aims to maximize the focality of the E-field while penalizing
deviations from the target location and orientation. It leverages the JAX
library for high-performance computation and the evosax library for the
evolutionary optimization algorithm.
"""

import json
from functools import partial
from typing import Callable, Dict, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from evosax.algorithms import DifferentialEvolution as DE
from plotly.subplots import make_subplots
from scipy.io import loadmat
from scipy.stats.qmc import Sobol

# --- Type Aliases for Clarity ---
Array = jnp.ndarray
NpArray = np.ndarray

# --- JAX-Compatible Data Structures ---
class Mesh(NamedTuple):
    """Represents a 3D mesh structure."""
    vertices: Array
    faces: Array
    vertex_normals: Array
    average_normal: Array


class OptimizerSettings(NamedTuple):
    """Configuration for the Differential Evolution optimizer."""
    popsize: int
    num_generations: int
    crossover_rate: float
    differential_weight: float
    dist_constraint_m: float
    angle_constraint_deg: float
    w_focality: float
    w_penalty: float
    k_penalty: float
    t_penalty: float


class Stimulator(NamedTuple):
    """Properties of the TMS stimulator device."""
    max_current_slope: Array
    polarity: Array


class OptimizationTarget(NamedTuple):
    """Defines the target for the optimization."""
    position: Array
    direction: Array


# --- Core Mathematical & Projection Functions (JAX Compatible) ---

def project_and_flatten(
    vertices: Array, normal: Array, origin: Array
) -> Tuple[Array, Array]:
    """Projects 3D vertices onto a plane and flattens them to 2D.

    This function is a JAX-based equivalent of the MATLAB `projectAndFlatten`
    utility. It creates a 2D coordinate system on the plane defined by the
    normal and origin.

    Args:
        vertices: An array of 3D vertices to project (shape: [N, 3]).
        normal: The normal vector of the projection plane (shape: [3,]).
        origin: The origin of the projection plane (shape: [3,]).

    Returns:
        A tuple containing:
        - The 2D coordinates of the projected vertices (shape: [N, 2]).
        - The orthonormal basis (U, W) of the plane (shape: [3, 2]).
    """
    v_shifted = vertices - origin
    dot_products = v_shifted @ normal
    p_shifted = v_shifted - (dot_products[:, None]) * normal

    # Find a robust orthonormal basis {U, W} for the plane.
    # This logic avoids picking a basis vector parallel to the normal.
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


def _create_objective_function(
    normalized_efield_set: Array,
    efield_basis: Array,
    mesh: Mesh,
    settings: OptimizerSettings,
) -> Callable[[Array, OptimizationTarget], Dict[str, Array]]:
    """Creates and JIT-compiles the objective function for the optimizer.

    This function defines the core logic for evaluating a population of
    candidate solutions against the optimization target.
    """

    def _pos_to_idx(pos: Array, vertices: Array) -> Array:
        """Finds the index of the vertex closest to a given 3D position."""
        return jnp.argmin(jnp.sum((vertices - pos) ** 2, axis=1))

    def _e_to_mag(e_field: Array) -> Tuple[Array, Array, Array]:
        """Calculates the magnitude of the E-field at each vertex."""
        e_mag = jnp.linalg.norm(e_field, axis=1)
        e_max_idx = jnp.argmax(e_mag)
        e_max = e_mag[e_max_idx]
        return e_mag, e_max, e_max_idx

    def _get_stimulated_target(
        e_field: Array, vertices_2d: Array
    ) -> Tuple[Array, Array]:
        """Determines the realized stimulation location (2D) and direction (3D)."""
        e_mag, e_max, _ = _e_to_mag(e_field)
        e_mag_n = e_mag / (e_max + 1e-8)

        # Calculate the Weighted Center of Gravity (WCOG) in 2D.
        weights = (e_mag_n**10)[:, jnp.newaxis]
        loc_2d = jnp.sum(weights * vertices_2d, axis=0) / (
            jnp.sum(weights) + 1e-8
        )

        # Find the 3D direction at the vertex closest to the 2D WCOG.
        loc_idx = jnp.argmin(jnp.sum((vertices_2d - loc_2d) ** 2, axis=1))
        direction = e_field[loc_idx, :]
        dir_norm = jnp.linalg.norm(direction)
        dir_normalized = jnp.where(
            dir_norm > 1e-8, direction / dir_norm, jnp.zeros_like(direction)
        )
        return loc_2d, dir_normalized

    def _calculate_vector_angle_deg(
        v1: Array, v2: Array, eps: float = 1e-8
    ) -> Array:
        """Calculates the angle between two vectors in degrees."""
        v1_u = v1 / (jnp.linalg.norm(v1) + eps)
        v2_u = v2 / (jnp.linalg.norm(v2) + eps)
        dot_product = jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0)
        angle_rad = jnp.arccos(dot_product)
        return jnp.rad2deg(angle_rad)

    def _single_objective_fun(
        population: Array, target: OptimizationTarget
    ) -> Dict[str, Array]:
        """Calculates the objective for one member of the population."""
        # --- 1. Project Mesh onto a 2D Plane Centered at the Target ---
        dist_from_target = jnp.linalg.norm(
            mesh.vertices - target.position, axis=1
        )
        # Use a local neighborhood of vertices to define the projection plane
        neighborhood_mask = dist_from_target < 0.03
        plane_normal = jnp.mean(
            mesh.vertex_normals, where=neighborhood_mask[:, None], axis=0
        )
        plane_normal /= jnp.linalg.norm(plane_normal) + 1e-8

        mesh_vertices_2d, _ = project_and_flatten(
            mesh.vertices, plane_normal, target.position
        )
        # In the new 2D system, the target's position is the origin.
        target_pos_2d = jnp.zeros(2)

        # --- 2. Calculate Target Direction in the 2D Plane ---
        closest_vertex_idx = _pos_to_idx(target.position, mesh.vertices)
        projection_basis = efield_basis[closest_vertex_idx]
        target_dir_2d = target.direction @ projection_basis

        # --- 3. Compute Realized E-field and Stimulation Target ---
        e_combined = jnp.einsum(
            "c,cvd->vd", population, normalized_efield_set
        )
        loc_realized_2d, dir_realized_3d = _get_stimulated_target(
            e_combined, mesh_vertices_2d
        )
        dir_realized_2d = dir_realized_3d @ projection_basis

        # --- 4. Calculate Location and Angle Errors ---
        loc_error = jnp.linalg.norm(loc_realized_2d - target_pos_2d)
        angle_error = _calculate_vector_angle_deg(
            target_dir_2d, dir_realized_2d
        )

        # --- 5. Calculate Penalty Term (Softplus) ---
        loc_error_norm = loc_error / settings.dist_constraint_m
        angle_error_norm = angle_error / settings.angle_constraint_deg
        combined_error = loc_error_norm + angle_error_norm

        # Numerically stable softplus to avoid overflow
        x = settings.k_penalty * (combined_error - settings.t_penalty)
        penalty_term = jnp.where(
            x > 30.0, x, jnp.log(1.0 + jnp.exp(x))
        )

        # --- 6. Calculate Focality Objective ---
        e_mag, e_max, _ = _e_to_mag(e_combined)
        e_mag_n = e_mag / (e_max + 1e-8)
        focality_objective = jnp.mean(e_mag_n**2)

        # --- 7. Combine into Final Fitness Score ---
        total_fitness = (settings.w_focality * focality_objective) + (
            settings.w_penalty * penalty_term
        )

        return {
            "fitness": total_fitness,
            "focality": focality_objective,
            "penalty": penalty_term,
            "loc_error": loc_error,
            "angle_error": angle_error,
        }

    # Vectorize the function to evaluate the entire population in parallel
    return jax.jit(jax.vmap(_single_objective_fun, in_axes=(0, None)))


@partial(
    jax.jit,
    static_argnames=(
        "strategy",
        "strategy_params",
        "jitted_objective",
    ),
)
def _optimizer_step(
    carry: Tuple,
    _: None,
    strategy: DE,
    strategy_params: Dict,
    jitted_objective: Callable,
) -> Tuple[Tuple, Dict]:
    """Executes one step of the Differential Evolution optimization."""
    state, key, target = carry
    key, key_ask, key_tell = jax.random.split(key, 3)

    population, state = strategy.ask(key_ask, state, strategy_params)
    population = jnp.clip(population, -1.0, 1.0)

    eval_output = jitted_objective(population, target)
    state, strategy_metrics = strategy.tell(
        key_tell, population, eval_output["fitness"], state, strategy_params
    )

    # Log metrics from the best individual in the current generation
    best_idx = jnp.argmin(eval_output["fitness"])
    log_metrics = {
        "best_fitness": strategy_metrics["best_fitness"],
        "focality": eval_output["focality"][best_idx],
        "penalty": eval_output["penalty"][best_idx],
        "loc_error": eval_output["loc_error"][best_idx],
        "angle_error": eval_output["angle_error"][best_idx],
    }

    return (state, key, target), log_metrics


# --- Main Optimizer Class ---
class StimulusOptimizer:
    """Manages the TMS stimulus optimization workflow."""

    # --- Constants ---
    MSO_CONSTRAINT = 0.5  # Maximum Stimulator Output constraint

    DEFAULT_SETTINGS = OptimizerSettings(
        popsize=256,
        num_generations=200,
        crossover_rate=0.0755,
        differential_weight=0.9871,
        dist_constraint_m=0.001,
        angle_constraint_deg=5.0,
        w_focality=1.0,
        w_penalty=1.0,
        k_penalty=5.0,
        t_penalty=1.0,
    )

    def __init__(self, settings: OptimizerSettings = None):
        """Initializes the StimulusOptimizer.

        Args:
            settings: An OptimizerSettings object. If None, default settings
                      are used.
        """
        self.settings = settings or self.DEFAULT_SETTINGS
        self.jax_device = jax.devices()[0]
        print(f"Using JAX device: {self.jax_device}")

        # --- Attributes to be loaded ---
        self.stimulator: Stimulator = None
        self.mesh: Mesh = None
        self.efield_set: Array = None
        self.efield_basis: Array = None
        self.n_coils: int = None

        # --- Internal state ---
        self._strategy: DE = None
        self._strategy_params: Dict = None
        self._jitted_objective: Callable = None
        self._jitted_step: Callable = None
        self._sobol_sampler: Sobol = None
        self.loss_components: Dict[str, NpArray] = None

    def load_stimulator(self, filename: str):
        """Loads stimulator properties from a JSON file.

        Args:
            filename: Path to the JSON file containing stimulator data.
        """
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            # Ensure required keys are present and convert to JAX arrays
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

            self.stimulator = Stimulator(
                max_current_slope=max_current_slope, polarity=polarity
            )
            print(f"Successfully loaded stimulator from {filename}.")

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise IOError(f"Error reading or processing stimulator file: {e}")

    def load_efield_model(self, filename: str):
        """Loads E-field and mesh data from a .mat file and prepares it."""
        if self.stimulator is None:
            raise RuntimeError(
                "Stimulator must be loaded before the E-field model."
            )

        try:
            efile = loadmat(filename)
            efield_set = jnp.asarray(
                np.stack(efile["Efield"].flatten()), dtype=jnp.float32
            )
        except (FileNotFoundError, KeyError) as e:
            raise IOError(f"Failed to load or parse .mat file: {e}")

        # Normalize E-field by max current slope
        normalized_efield_set = (
            efield_set * self.stimulator.max_current_slope.reshape(-1, 1, 1)
        )
        self.efield_set = jax.device_put(normalized_efield_set, self.jax_device)
        self.n_coils = self.efield_set.shape[0]
        self._sobol_sampler = Sobol(d=self.n_coils, scramble=True)

        # --- Load and process mesh data ---
        mesh_data = {
            "vertices": jnp.asarray(
                efile["smesh"]["p"][0][0], dtype=jnp.float32
            ),
            # Convert to 0-based indexing for Python
            "faces": jnp.asarray(
                efile["smesh"]["e"][0][0].astype(np.int32) - 1, dtype=jnp.int32
            ),
        }
        mesh_gpu = {
            k: jax.device_put(v, self.jax_device)
            for k, v in mesh_data.items()
        }

        # --- Calculate vertex and average normals ---
        vertex_normals = self._calculate_vertex_normals(
            mesh_gpu["vertices"], mesh_gpu["faces"]
        )
        mean_normal = jnp.nanmean(vertex_normals, axis=0)
        norm_mean_normal = jnp.linalg.norm(mean_normal)
        avg_normal = jnp.where(
            norm_mean_normal > 1e-8,
            mean_normal / norm_mean_normal,
            jnp.zeros_like(mean_normal),
        )

        self.mesh = Mesh(
            vertices=mesh_gpu["vertices"],
            faces=mesh_gpu["faces"],
            vertex_normals=vertex_normals,
            average_normal=avg_normal,
        )

        # --- Pre-compute E-field basis for directional calculations ---
        self._compute_efield_basis()

        # --- Create the JIT-compiled objective function ---
        self._jitted_objective = _create_objective_function(
            self.efield_set, self.efield_basis, self.mesh, self.settings
        )
        print(f"Successfully loaded and prepared E-field model from {filename}.")

    def run(
        self, target_position: NpArray, target_direction: NpArray
    ) -> Tuple[NpArray, NpArray, NpArray]:
        """Runs the optimization process for a given target.

        Args:
            target_position: A numpy array (3,) for the target 3D position.
            target_direction: A numpy array (3,) for the target 3D direction.

        Returns:
            A tuple containing:
            - The optimal coil weights (current slopes).
            - The final location error in meters.
            - The final angle error in degrees.
        """
        if self._jitted_objective is None:
            raise RuntimeError(
                "E-field model must be loaded before running optimization."
            )

        self._setup_strategy()
        jitted_step = self._get_or_create_jitted_step()

        key = jax.random.key(0)
        target = OptimizationTarget(
            position=jax.device_put(
                jnp.asarray(target_position, dtype=jnp.float32),
                self.jax_device,
            ),
            direction=jax.device_put(
                jnp.asarray(target_direction, dtype=jnp.float32),
                self.jax_device,
            ),
        )

        # --- Initialize Population ---
        initial_population = (
            self._sobol_sampler.random(n=self.settings.popsize) * 2 - 1
        )
        initial_population = self._constrain_energy(
            initial_population.astype(np.float32)
        )
        initial_population = jax.device_put(initial_population, self.jax_device)

        # --- Initialize Optimizer State ---
        initial_eval = self._jitted_objective(initial_population, target)
        key, subkey = jax.random.split(key)
        state = self._strategy.init(
            subkey,
            initial_population,
            initial_eval["fitness"],
            self._strategy_params,
        )

        # --- Run Optimization Loop ---
        initial_carry = (state, key, target)
        scan_xs = jnp.arange(self.settings.num_generations)
        final_carry, metrics_log = jax.lax.scan(
            jitted_step, initial_carry, scan_xs
        )

        final_state, _, _ = final_carry
        final_state.best_solution.block_until_ready()

        self.loss_components = {k: np.array(v) for k, v in metrics_log.items()}

        # Re-evaluate the single best solution to get its final errors
        best_solution_reshaped = final_state.best_solution[jnp.newaxis, :]
        final_eval_output = self._jitted_objective(best_solution_reshaped, target)

        final_loc_error = np.array(final_eval_output["loc_error"][0])
        final_angle_error = np.array(final_eval_output["angle_error"][0])
        best_solution = np.array(final_state.best_solution)

        # Apply final energy constraint and scale to physical units
        constrained_solution = self._constrain_energy(best_solution)
        final_weights = (
            constrained_solution * self.stimulator.max_current_slope
        )

        return final_weights, final_loc_error, final_angle_error

    # --- Private Helper Methods ---

    def _get_or_create_jitted_step(self) -> Callable:
        """Lazily creates and JIT-compiles the optimizer step function."""
        if self._jitted_step is None:
            self._jitted_step = partial(
                _optimizer_step,
                strategy=self._strategy,
                strategy_params=self._strategy_params,
                jitted_objective=self._jitted_objective,
            )
        return self._jitted_step

    def _setup_strategy(self):
        """Initializes the evosax Differential Evolution strategy."""
        dummy_solution = jnp.ones(self.n_coils) * (self.MSO_CONSTRAINT * 0.9)
        self._strategy = DE(
            population_size=self.settings.popsize, solution=dummy_solution
        )
        self._strategy_params = self._strategy.default_params.replace(
            crossover_rate=self.settings.crossover_rate,
            differential_weight=self.settings.differential_weight,
        )

    def _constrain_energy(self, x: Array) -> Array:
        """Ensures a stimulus does not exceed the maximum energy limit."""
        x = jnp.asarray(x, dtype=jnp.float32)
        energy_limit = jnp.sum((jnp.ones(self.n_coils) * self.MSO_CONSTRAINT) ** 2)

        is_1d = x.ndim == 1
        if is_1d:
            x = x[jnp.newaxis, :]

        solution_energy = jnp.sum(x**2, axis=-1)
        violation_mask = solution_energy > energy_limit

        violating_solutions = x[violation_mask]
        violating_energy = solution_energy[violation_mask]
        scale_factors = jnp.sqrt(violating_energy / energy_limit)

        # Scale down violating solutions just below the limit
        corrected_solutions = (
            violating_solutions / scale_factors[:, jnp.newaxis]
        ) * 0.999
        x = x.at[violation_mask].set(corrected_solutions)

        return x.squeeze(axis=0) if is_1d else x

    def _compute_efield_basis(self):
        """Computes a 2D basis for the E-field at each vertex using SVD."""
        e_reshaped = self.efield_set.transpose(1, 0, 2)  # [V, C, D]
        means = e_reshaped.mean(axis=1)
        e_centered = e_reshaped - means[:, None, :]

        # Perform SVD per vertex to find principal components of E-field vectors
        _, _, vh = jax.vmap(jnp.linalg.svd)(e_centered)
        v_basis = vh.transpose(0, 2, 1)  # [V, D, C]
        v_top2 = v_basis[:, :, :2]  # [V, 3, 2]

        # Ensure consistent orientation of the basis vectors
        plane_normals = jnp.cross(v_top2[..., 0], v_top2[..., 1], axis=-1)
        dot_products = jnp.sum(
            plane_normals * self.mesh.average_normal, axis=-1
        )
        flip_mask = dot_products < 0
        flipper = jnp.array([1.0, -1.0])
        self.efield_basis = v_top2 * jnp.where(
            flip_mask[:, None, None], flipper[None, None, :], 1.0
        )

    @staticmethod
    def _calculate_vertex_normals(vertices: Array, faces: Array) -> Array:
        """Calculates vertex normals for a triangulated 3D mesh using JAX."""
        v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        face_normals = jnp.cross(v1, v2)

        # Safe normalization of face normals
        face_norms = jnp.linalg.norm(face_normals, axis=1, keepdims=True)
        safe_face_normals = jnp.where(
            face_norms > 1e-8,
            face_normals / face_norms,
            jnp.zeros_like(face_normals),
        )

        # Accumulate face normals at each vertex
        vertex_normals = jnp.zeros_like(vertices)
        vertex_normals = vertex_normals.at[faces[:, 0]].add(safe_face_normals)
        vertex_normals = vertex_normals.at[faces[:, 1]].add(safe_face_normals)
        vertex_normals = vertex_normals.at[faces[:, 2]].add(safe_face_normals)

        # Safe normalization of vertex normals
        norms = jnp.linalg.norm(vertex_normals, axis=1, keepdims=True)
        safe_vertex_normals = jnp.where(
            norms > 1e-8,
            vertex_normals / norms,
            jnp.zeros_like(vertex_normals),
        )

        # Handle vertices with zero-norm normals (e.g., isolated vertices)
        is_zero_norm = (norms < 1e-8).squeeze()
        default_normal = jnp.array([0.0, 0.0, 1.0])
        return jnp.where(
            is_zero_norm[:, None], default_normal, safe_vertex_normals
        )


# --- Visualization Module ---
# Note: In a larger project, this would be in a separate file.


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

    # Find 3D direction
    loc_idx = jnp.argmin(jnp.sum((vertices_2d - loc_2d) ** 2, axis=1))
    direction_3d = e_field[loc_idx, :]
    dir_norm = jnp.linalg.norm(direction_3d)
    dir_normalized_3d = jnp.where(
        dir_norm > 1e-8, direction_3d / dir_norm, jnp.zeros_like(direction_3d)
    )

    # Convert 2D location back to 3D
    loc_3d = (basis_2d @ loc_2d) + target_pos

    return loc_3d, dir_normalized_3d


def visualize_solution_plotly(
    optimizer: StimulusOptimizer,
    solution_weights: NpArray,
    optimization_target: Tuple[NpArray, NpArray],
):
    """Creates an interactive 3D plot of the optimization result using Plotly.

    Args:
        optimizer: The completed StimulusOptimizer instance.
        solution_weights: The final optimized coil weights.
        optimization_target: A tuple of (position, direction).
    """
    target_pos, target_dir = optimization_target
    target_pos_jax = jnp.asarray(target_pos)

    # Normalize weights to range [-1, 1] for E-field calculation
    normalized_weights = solution_weights / optimizer.stimulator.max_current_slope
    e_combined = jnp.einsum(
        "c,cvd->vd", normalized_weights, optimizer.efield_set
    )
    e_mag, e_max, _ = _e_to_mag(e_combined)

    realized_pos, realized_dir = _get_stimulated_target_3d(
        e_combined, optimizer.mesh, target_pos_jax
    )

    # --- Create Plotly Figure ---
    fig = go.Figure(
        go.Mesh3d(
            x=np.array(optimizer.mesh.vertices[:, 0]),
            y=np.array(optimizer.mesh.vertices[:, 1]),
            z=np.array(optimizer.mesh.vertices[:, 2]),
            i=np.array(optimizer.mesh.faces[:, 0]),
            j=np.array(optimizer.mesh.faces[:, 1]),
            k=np.array(optimizer.mesh.faces[:, 2]),
            intensity=np.array(e_mag / (e_max + 1e-8)),
            colorscale="Viridis",
            colorbar_title="Normalized E-Field",
            name="Brain Mesh",
        )
    )

    arrow_scale = np.mean(np.ptp(np.array(optimizer.mesh.vertices), axis=0)) * 0.2
    # Add Target Arrow
    fig.add_trace(
        go.Cone(
            x=[target_pos[0]],
            y=[target_pos[1]],
            z=[target_pos[2]],
            u=[target_dir[0]],
            v=[target_dir[1]],
            w=[target_dir[2]],
            sizeref=arrow_scale,
            showscale=False,
            colorscale=[[0, "red"], [1, "red"]],
            anchor="tip",
            name="Target",
        )
    )
    # Add Realized Arrow
    fig.add_trace(
        go.Cone(
            x=[realized_pos[0]],
            y=[realized_pos[1]],
            z=[realized_pos[2]],
            u=[realized_dir[0]],
            v=[realized_dir[1]],
            w=[realized_dir[2]],
            sizeref=arrow_scale,
            showscale=False,
            colorscale=[[0, "cyan"], [1, "cyan"]],
            anchor="tip",
            name="Realized",
        )
    )

    fig.update_layout(
        title="Optimization Result: E-Field and Direction",
        legend=dict(x=0.8, y=0.9),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # --- Set Camera ---
    mesh_center = np.mean(optimizer.mesh.vertices, axis=0)
    distance_factor = np.max(np.std(optimizer.mesh.vertices, axis=0)) * 100
    camera_pos = mesh_center + np.array(optimizer.mesh.average_normal) * distance_factor
    camera = {
        "eye": {
            "x": camera_pos[0].item(),
            "y": camera_pos[1].item(),
            "z": camera_pos[2].item(),
        },
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {
            "x": mesh_center[0].item(),
            "y": mesh_center[1].item(),
            "z": mesh_center[2].item(),
        },
    }
    fig.update_scenes(
        camera=camera,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    )

    fig.show()


def visualize_loss_components(optimizer: StimulusOptimizer):
    """Visualizes the convergence of the loss and its components.

    Args:
        optimizer: The completed StimulusOptimizer instance.
    """
    loss_data = optimizer.loss_components
    if not loss_data or "best_fitness" not in loss_data:
        print(
            "Loss components not found. Please run the optimizer first."
        )
        return

    generations = np.arange(len(loss_data["best_fitness"]))

    # Safely calculate the ratio of weighted components
    focality = loss_data["focality"]
    penalty = loss_data["penalty"]
    ratio = focality / (penalty + 1e-9)  # Avoid division by zero

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Total Loss Convergence",
            "Focality / Penalty Ratio",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=generations,
            y=loss_data["best_fitness"],
            mode="lines",
            name="Total Loss",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=generations, y=ratio, mode="lines", name="Focality/Penalty Ratio"
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_yaxes(title_text="Loss Value", row=1, col=1)
    fig.update_yaxes(title_text="Ratio (log scale)", type="log", row=2, col=1)
    fig.update_layout(
        height=800, title_text="Optimization Loss Analysis", showlegend=False
    )

    fig.show()
