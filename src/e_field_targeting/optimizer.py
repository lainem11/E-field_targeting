"""
Performs stimulus optimization for transcranial magnetic stimulation (TMS)
using Differential Evolution.

This module defines the `StimulusOptimizer` class, which encapsulates the logic
for setting up and running an optimization process to find the optimal coil
currents for a desired stimulation target. This is considered the internal
"core" of the optimization engine.
"""
from functools import partial
from typing import Callable, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms import DifferentialEvolution as DE
from scipy.stats.qmc import Sobol

# --- Import from local package modules ---
from .utils import project_and_flatten

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


class EFieldModel(NamedTuple):
    """Represents the pre-processed E-field data and mesh."""

    mesh: Mesh
    efield_set: Array
    efield_basis: Array


class OptimizerSettings(NamedTuple):
    """Configuration for the Differential Evolution optimizer."""

    popsize: int
    num_generations: int
    crossover_rate: float
    differential_weight: float
    dist_constraint_m: float
    angle_constraint_deg: float
    max_dist_constraint_m: float
    w_focality: float
    w_penalty: float
    k_penalty: float
    t_penalty: float
    w_energy: float
    k_energy: float
    energy_limit_fn: Callable[[int], float]


class Stimulator(NamedTuple):
    """Properties of the TMS stimulator device."""

    max_current_slope: Array
    polarity: Array


class OptimizationTarget(NamedTuple):
    """Defines the target for the optimization."""

    position: Array
    direction: Array
    intensity: Array


# --- Objective Function and Optimizer Step ---


def _create_objective_function(
    max_efield_set: Array,
    efield_basis: Array,
    mesh: Mesh,
    settings: OptimizerSettings,
) -> Callable[[Array, OptimizationTarget], Dict[str, Array]]:
    """Creates and JIT-compiles the objective function for the optimizer."""

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

        weights = (e_mag_n**10)[:, jnp.newaxis]
        loc_2d = jnp.sum(weights * vertices_2d, axis=0) / (jnp.sum(weights) + 1e-8)

        loc_idx = jnp.argmin(jnp.sum((vertices_2d - loc_2d) ** 2, axis=1))
        direction = e_field[loc_idx, :]
        dir_norm = jnp.linalg.norm(direction)
        dir_normalized = jnp.where(
            dir_norm > 1e-8, direction / dir_norm, jnp.zeros_like(direction)
        )
        return loc_2d, dir_normalized

    def _calculate_vector_angle_deg(v1: Array, v2: Array, eps: float = 1e-8) -> Array:
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
        dist_from_target = jnp.linalg.norm(mesh.vertices - target.position, axis=1)
        neighborhood_mask = dist_from_target < 0.03
        plane_normal = jnp.mean(
            mesh.vertex_normals, where=neighborhood_mask[:, None], axis=0
        )
        plane_normal /= jnp.linalg.norm(plane_normal) + 1e-8

        mesh_vertices_2d, _ = project_and_flatten(
            mesh.vertices, plane_normal, target.position
        )
        target_pos_2d = jnp.zeros(2)

        closest_vertex_idx = _pos_to_idx(target.position, mesh.vertices)
        projection_basis = efield_basis[closest_vertex_idx]
        target_dir_2d = target.direction @ projection_basis

        e_combined = jnp.einsum("c,cvd->vd", population, max_efield_set)
        loc_realized_2d, dir_realized_3d = _get_stimulated_target(
            e_combined, mesh_vertices_2d
        )
        dir_realized_2d = dir_realized_3d @ projection_basis

        e_mag, e_max, e_max_idx = _e_to_mag(e_combined)
        e_mag_n = e_mag / (e_max + 1e-8)
        focality_objective = jnp.mean(e_mag_n**2)

        scaling_factor = target.intensity / (e_max + 1e-8)
        solution_energy = jnp.sum((scaling_factor * population) ** 2)
        energy_limit = settings.energy_limit_fn(max_efield_set.shape[0])

        energy_violation = jax.nn.relu(solution_energy - energy_limit)
        x_energy = settings.k_energy * energy_violation
        energy_penalty = jnp.where(
            x_energy > 30.0, x_energy, jnp.log(1.0 + jnp.exp(x_energy))
        )

        loc_error = jnp.linalg.norm(loc_realized_2d - target_pos_2d)
        angle_error = _calculate_vector_angle_deg(target_dir_2d, dir_realized_2d)
        e_max_distance = jnp.linalg.norm(mesh_vertices_2d[e_max_idx])

        loc_error_norm = loc_error / settings.dist_constraint_m
        angle_error_norm = angle_error / settings.angle_constraint_deg
        e_max_error_norm = e_max_distance / settings.max_dist_constraint_m

        combined_error = loc_error_norm + angle_error_norm + e_max_error_norm

        x_penalty = settings.k_penalty * (combined_error - settings.t_penalty)
        accuracy_penalty = jnp.where(
            x_penalty > 30.0, x_penalty, jnp.log(1.0 + jnp.exp(x_penalty))
        )

        focality_term = settings.w_focality * focality_objective
        accuracy_term = settings.w_penalty * accuracy_penalty
        energy_term = settings.w_energy * energy_penalty
        total_fitness = focality_term + accuracy_term + energy_term

        return {
            "fitness": total_fitness,
            "focality": focality_term,
            "accuracy": accuracy_term,
            "energy": energy_term,
            "loc_error": loc_error,
            "angle_error": angle_error,
        }

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

    best_idx = jnp.argmin(eval_output["fitness"])
    log_metrics = {
        "best_fitness": strategy_metrics["best_fitness"],
        "focality": eval_output["focality"][best_idx],
        "accuracy": eval_output["accuracy"][best_idx],
        "energy": eval_output["energy"][best_idx],
        "loc_error": eval_output["loc_error"][best_idx],
        "angle_error": eval_output["angle_error"][best_idx],
    }

    return (state, key, target), log_metrics


# --- Main Optimizer Class ---
class StimulusOptimizer:
    """Manages the TMS stimulus optimization workflow."""

    MSO_CONSTRAINT = 0.5
    DEFAULT_SETTINGS = OptimizerSettings(
        popsize=256,
        num_generations=200,
        crossover_rate=0.0755,
        differential_weight=0.9871,
        dist_constraint_m=0.001,
        angle_constraint_deg=5.0,
        max_dist_constraint_m=0.01,
        w_focality=1.0,
        w_penalty=1.0,
        k_penalty=5.0,
        t_penalty=1.0,
        w_energy=1.0,
        k_energy=5.0,
        energy_limit_fn=lambda n_coils, mso=MSO_CONSTRAINT: (
            np.sum((np.ones(n_coils) * mso) ** 2)
        ),
    )

    def __init__(
        self,
        stimulator: Stimulator,
        mesh: Mesh,
        efield_set: Array,
        efield_basis: Array,
        settings: OptimizerSettings = None,
    ):
        """
        Initializes the StimulusOptimizer with pre-loaded data.
        """
        self.settings = settings or self.DEFAULT_SETTINGS
        self.jax_device = jax.devices()[0]

        self.stimulator = stimulator
        self.mesh = mesh
        self.efield_set = efield_set
        self.efield_basis = efield_basis
        self.n_coils = self.efield_set.shape[0]

        self._strategy: DE = None
        self._strategy_params: Dict = None
        self._sobol_sampler = Sobol(d=self.n_coils, scramble=True)
        self.loss_components: Dict[str, NpArray] = None

        self._jitted_objective = _create_objective_function(
            self.efield_set, self.efield_basis, self.mesh, self.settings
        )
        self._jitted_step = None  # Lazily created

    def run(
        self,
        target_position: NpArray,
        target_direction: NpArray,
        target_intensity: float,
    ) -> Tuple[NpArray, NpArray, NpArray]:
        """Runs the optimization process for a given target."""
        self._setup_strategy()
        jitted_step = self._get_or_create_jitted_step()

        key = jax.random.key(0)
        target = OptimizationTarget(
            position=jax.device_put(
                jnp.asarray(target_position, dtype=jnp.float32), self.jax_device
            ),
            direction=jax.device_put(
                jnp.asarray(target_direction, dtype=jnp.float32), self.jax_device
            ),
            intensity=jax.device_put(
                jnp.asarray(target_intensity, dtype=jnp.float32), self.jax_device
            ),
        )

        initial_population = (
            self._sobol_sampler.random(n=self.settings.popsize) * 2 - 1
        ).astype(np.float32)
        initial_population = self._constrain_energy(initial_population)
        initial_population = jax.device_put(initial_population, self.jax_device)

        initial_eval = self._jitted_objective(initial_population, target)
        key, subkey = jax.random.split(key)
        state = self._strategy.init(
            subkey, initial_population, initial_eval["fitness"], self._strategy_params
        )

        initial_carry = (state, key, target)
        scan_xs = jnp.arange(self.settings.num_generations)
        final_carry, metrics_log = jax.lax.scan(jitted_step, initial_carry, scan_xs)

        final_state, _, _ = final_carry
        final_state.best_solution.block_until_ready()
        self.loss_components = {k: np.array(v) for k, v in metrics_log.items()}

        best_solution_reshaped = final_state.best_solution[jnp.newaxis, :]
        final_eval_output = self._jitted_objective(best_solution_reshaped, target)
        final_loc_error = np.array(final_eval_output["loc_error"][0])
        final_angle_error = np.array(final_eval_output["angle_error"][0])
        best_solution = final_state.best_solution

        e_combined = jnp.einsum("c,cvd->vd", best_solution, self.efield_set)
        e_max = jnp.max(jnp.linalg.norm(e_combined, axis=1))
        scaling_factor = target_intensity / (e_max + 1e-8)
        final_weights = (
            best_solution * scaling_factor * self.stimulator.max_current_slope
        )

        return final_weights, final_loc_error, final_angle_error

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
        corrected_solutions = (
            violating_solutions / scale_factors[:, jnp.newaxis]
        ) * 0.999
        x = x.at[violation_mask].set(corrected_solutions)
        return x.squeeze(axis=0) if is_1d else x

