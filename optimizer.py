from functools import partial
import jax
import jax.numpy as jnp
from scipy.stats.qmc import Sobol
from evosax.algorithms import DifferentialEvolution as DE
from typing import Callable, Dict, Tuple
import numpy as np
from scipy.io import loadmat
import json

# Define type aliases for clarity
jaxArray = jnp.ndarray
npArray = np.ndarray

def _create_objective_function(normalized_efield_set: jaxArray, efield_basis: jaxArray, efield_basis_means: jaxArray, mesh_vertices: jaxArray, optimizer_settings: Dict) -> Callable[[jaxArray, Tuple], jaxArray]:
    """
    Creates and JIT-compiles the objective function for the optimizer.
    """
    def pos_to_ind(pos: jaxArray, vertices: jaxArray) -> jaxArray:
        """
        Finds the index of the vertex closest to a given 3D position.
        
        Args:
            pos: Target position, shape (3,).
            vertices: Mesh vertices, shape (n_vertices, 3).
            
        Returns:
            The integer index of the closest vertex.
        """
        # Placeholder: Replace with your implementation.
        # This is a common implementation using squared Euclidean distance for efficiency.
        return jnp.argmin(jnp.sum((vertices - pos)**2, axis=1))

    def e_to_mag(E: jaxArray) -> Tuple[jaxArray, jaxArray, jaxArray]:
        """
        Calculates the magnitude of the E-field at each vertex.
        
        Args:
            E: Combined E-field, shape (n_vertices, 3).
            
        Returns:
            A tuple containing:
            - E_mag: The magnitude at each vertex, shape (n_vertices,).
            - E_max: The maximum magnitude value (scalar).
            - E_max_ind: The index of the vertex with the maximum magnitude.
        """
        # Placeholder: Replace with your implementation.
        E_mag = jnp.linalg.norm(E, axis=1)
        E_max_ind = jnp.argmax(E_mag)
        E_max = E_mag[E_max_ind]
        return E_mag, E_max, E_max_ind

    def stimulated_target(E: jaxArray, vertices: jaxArray) -> Tuple[jaxArray, jaxArray]:
        """
        Determines the realized stimulation location and direction from the E-field.
        
        Args:
            E: Combined E-field, shape (n_vertices, 3).
            vertices: Mesh vertices, shape (n_vertices, 3).
            
        Returns:
            A tuple containing:
            - loc: The realized stimulation location, shape (3,).
            - dir: The realized stimulation direction (normalized), shape (3,).
        """
        # Placeholder: Replace with your implementation based on 'Max' or 'WCOG'.
        # This example uses the 'Max' metric for simplicity.
        _, _, loc_i = e_to_mag(E)
        loc = vertices[loc_i, :]
        direction = E[loc_i, :]
        direction_norm = jnp.linalg.norm(direction)
        # Avoid division by zero for null vectors
        dir_normalized = jnp.where(direction_norm > 1e-8, direction / direction_norm, jnp.zeros_like(direction))
        return loc, dir_normalized

    def calculate_vector_angle(v1: jaxArray, v2: jaxArray, eps: float = 1e-8) -> jaxArray:
        """
        Calculates the angle between two vectors in degrees.
        """
        # Normalize the vectors
        v1_u = v1 / (jnp.linalg.norm(v1) + eps)
        v2_u = v2 / (jnp.linalg.norm(v2) + eps)
        
        # Calculate the dot product and clip to avoid numerical errors with acos
        dot_product = jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0)
        
        # Calculate angle in radians and convert to degrees
        angle_rad = jnp.arccos(dot_product)
        return jnp.rad2deg(angle_rad)

    def single_objective_fun(population: jaxArray, target: Tuple):
        """
        Calculates the objective function for a single optimizer population.

        Args:
            population: Candidate solution for optimized parameters (coil weights).
            target: Tuple of size two with elements:
                position: Array of E-field target coordinates.
                direction: Array of E-field target direction.
        """
        position, direction = target
        
        # 1. Find the 2D basis corresponding to the vertex closest to the target.
        # This replicates the MATLAB logic of creating a 2D subspace for direction analysis.
        closest_vertex_idx = pos_to_ind(position, mesh_vertices)
        V = efield_basis[closest_vertex_idx] # Shape: (3, 2)
        
        # 2. Project the target direction onto this 2D basis.
        target_dir_2d = direction @ V
        
        # 3. Calculate the combined E-field from the weighted sum of coil contributions.
        # 'c' is coils, 'v' is vertices, 'd' is spatial dimensions (3).
        E_combined = jnp.einsum('c,cvd->vd', population, normalized_efield_set)
        
        # 4. Determine the realized stimulation location and direction from the combined E-field.
        loc_realized, dir_realized = stimulated_target(E_combined, mesh_vertices)
        
        # 5. Project the realized E-field direction onto the same 2D basis for comparison.
        dir_realized_2d = dir_realized @ V
        
        # 6. Calculate location and angle errors (constraints).
        loc_error = jnp.linalg.norm(position - loc_realized)
        angle_error = calculate_vector_angle(target_dir_2d, dir_realized_2d)

        # 7. Define penalties for constraint violations.
        # The penalty is zero if the error is within the allowed constraint.
        loc_penalty = jnp.maximum(0, loc_error - optimizer_settings['dist_constraint'])
        angle_penalty = jnp.maximum(0, angle_error - optimizer_settings['angle_constraint'])
        
        # 8. Calculate the primary objective: focality.
        # This corresponds to the 'Focality' objective in the MATLAB script.
        E_mag, E_max, _ = e_to_mag(E_combined)
        
        # Add a small epsilon to E_max to prevent division by zero if E-field is all zeros.
        E_mag_n = E_mag / (E_max + 1e-8)
        focality_objective = jnp.mean(E_mag_n**2)
        
        # 9. Combine the objective with penalties to get the final fitness value.
        # The optimizer will seek to minimize this value.
        total_fitness = focality_objective + optimizer_settings['constraint_penalty'] * (loc_penalty + angle_penalty)
        
        return total_fitness


    return jax.jit(jax.vmap(single_objective_fun, in_axes=(0,None)))

def _optimizer_step(carry: Tuple, _: None, strategy: DE, strategy_params: Dict, jitted_objective: Callable) -> Tuple[Tuple, Dict]:
    """Executes one step of the Differential Evolution optimization.

    This function is designed to be used with `jax.lax.scan` for efficient,
    JIT-compiled looping over optimization generations.

    Args:
        carry: A tuple containing the optimizer state, random key, and static
               data (stimulation target specs).
        _: Placeholder for the scan's sequence data (unused).
        strategy: The evosax Differential Evolution strategy instance.
        strategy_params: Hyperparameters for the DE strategy.
        jitted_objective: The pre-compiled objective function.

    Returns:
        A tuple containing the updated carry and a dictionary of metrics for the step.
    """
    state, key, target = carry
    key, key_ask, key_tell = jax.random.split(key, 3)

    # Ask the optimizer for a new population of candidate solutions.
    population, state = strategy.ask(key_ask, state, strategy_params)
    population = jnp.clip(population, -1.0, 1.0) # Ensure solutions are within [-1, 1] range

    # Evaluate the fitness of the population.
    fitness = jitted_objective(population, target)
    # Tell the optimizer the results to update its state.
    state, metrics = strategy.tell(key_tell, population, fitness, state, strategy_params)

    return (state, key, target), metrics

class StimulusOptimizer:

    def __init__(self):
        self.jax_device = jax.devices()[0]
        self.MSO_constraint = 0.5
        self.energy_limit = None

        self.optimizer_settings = {
            'popsize': 256,
            'num_generations': 200,
            'constraint_penalty': 1e4, # Increased penalty to enforce constraints strongly
            'crossover_rate': 0.0755,
            'differential_weight': 0.9871,
            'dist_constraint': 0.001,  # in meters, from MATLAB default
            'angle_constraint': 5.0    # in degrees, from MATLAB default
        }

        self.n_coils = None
        self.energy_limit = None
        self.max_current_slope = None
        self.jitted_objective = None
        self.sobol_sampler = None
        self.losses = None
        self.efield_set = None
        self.efield_basis = None
        self.efield_basis_means = None
        self.mesh = None
        self.stimulator = None

        self.jitted_step = None

    @staticmethod
    def calculate_node_normals(vertices: jaxArray, triangles: jaxArray) -> jaxArray:
        """
        Calculate vertex normals for a triangulated 3D mesh using JAX.

        Args:
            vertices: (N, 3) array of vertex coordinates.
            triangles: (M, 3) array of triangle indices.

        Returns:
            (N, 3) array of normalized vertex normals.
        """
        # Compute face normals
        v1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
        v2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
        face_normals = jnp.cross(v1, v2)

        # Normalize face normals, handling potential zero-norm faces
        face_norms = jnp.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = jnp.where(face_norms > 1e-8, face_normals / face_norms, jnp.zeros_like(face_normals))

        # Accumulate face normals for each vertex
        vertex_normals = jnp.zeros_like(vertices)
        vertex_normals = vertex_normals.at[triangles[:, 0]].add(face_normals)
        vertex_normals = vertex_normals.at[triangles[:, 1]].add(face_normals)
        vertex_normals = vertex_normals.at[triangles[:, 2]].add(face_normals)

        # Normalize the vertex normals
        norms = jnp.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = jnp.where(norms > 1e-8, vertex_normals / norms, jnp.zeros_like(vertex_normals))
        
        # Replace any remaining zero-norm normals with a default vector [0, 0, 1]
        is_zero_norm = (norms < 1e-8).squeeze()
        default_normal = jnp.array([0., 0., 1.])
        vertex_normals = jnp.where(is_zero_norm[:, None], default_normal, vertex_normals)

        return vertex_normals

    def load_efield_model(self, filename: str):
        """Loads E-field and mesh data from a .mat file with a specific structure.

        Args:
            filename: The path to the .mat file.
        """
        assert self.stimulator is not None, "Specify stimulator before loading the efield model."
        Efile = loadmat(filename)
        efield_set = jnp.asarray(np.stack(Efile['Efield'].flatten()), dtype=jnp.float32)
        normalized_efield_set = efield_set * self.stimulator['max_current_slope'].reshape(-1,1,1)

        self.n_coils = normalized_efield_set.shape[0]
        self.sobol_sampler = Sobol(d=self.n_coils, scramble=True)

        # Extract mesh data and convert faces to be 0-indexed
        mesh = {
            'vertices': jnp.asarray(Efile['smesh']['p'][0][0],dtype=jnp.float32),
            'faces': jnp.asarray(Efile['smesh']['e'][0][0].astype(np.int32) - 1,dtype=jnp.int32),
        }
        
        # Move data to JAX device
        self.efield_set = jax.device_put(normalized_efield_set, self.jax_device)
        self.mesh = {key: jax.device_put(value, self.jax_device) for key, value in mesh.items()}

        ### Calculate average normal
        normals = self.calculate_node_normals(self.mesh['vertices'], self.mesh['faces'])
        self.mesh['vertex_normals'] = normals

        # Calculate the average normal vector
        mean_normal = jnp.nanmean(self.mesh['vertex_normals'], axis=0)
        
        norm_mean_normal = jnp.linalg.norm(mean_normal)
        self.mesh['average_normal'] = jnp.where(norm_mean_normal > 1e-8, mean_normal / norm_mean_normal, jnp.zeros_like(mean_normal))

        # Reshape to (n_vertices, n_coils, 3) for per-vertex PCA
        e_reshaped = self.efield_set.transpose(1, 0, 2)
        means = e_reshaped.mean(axis=1)  # Mean across coils for each vertex
        e_centered = e_reshaped - means[:, None, :]

        # Perform SVD on each vertex's (n_coils, 3) matrix to get principal components.
        # We use jax.vmap to efficiently apply svd across all vertices.
        _, _, Vh = jax.vmap(jnp.linalg.svd, in_axes=(0,))(e_centered)
        
        # Vh contains right singular vectors as rows. V = Vh.T contains them as columns.
        V = Vh.transpose(0, 2, 1)
        V_top2 = V[:, :, :2] # Top 2 principal components

        # Ensure the PCA plane normal aligns with the mesh surface normal.
        plane_normals = jnp.cross(V_top2[..., 0], V_top2[..., 1], axis=-1)
        dot_products = jnp.sum(plane_normals * self.mesh['average_normal'], axis=-1)
        flip_indices = dot_products < 0

        # Flip the second basis vector where necessary to align normals
        flipper = jnp.array([1.0, -1.0])
        V_flipped = V_top2 * jnp.where(flip_indices[:, None, None], flipper[None, None, :], 1.0)
        
        self.efield_basis = V_flipped
        self.efield_basis_means = means

        self.jitted_objective = _create_objective_function(self.efield_set,self.efield_basis,self.efield_basis_means,self.mesh['vertices'],self.optimizer_settings)
        
    def load_stimulator(self,filename: str):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading stimulator file: {e}")
            return None

        jnp_data = {}
        for key, value in data.items():
            try:
                jnp_data[key] = jnp.atleast_1d(jnp.asarray(value,dtype=jnp.float32))
            except (TypeError, ValueError):
                print(f"Warning: Could not convert key '{key}' to a jax array. Skipping.")

        # Calculate max_current_slope if not provided
        if 'max_current_slope' not in jnp_data:
            if 'max_voltage' in jnp_data and 'inductance' in jnp_data:
                jnp_data['max_current_slope'] = jnp_data['max_voltage'] / jnp_data['inductance']
            else:
                print("Error: Stimulator file must contain 'max_current_slope' or 'max_voltage' and 'inductance'.")
                return None

        # Assume positive polarity if not specified
        if 'polarity' not in jnp_data:
            jnp_data['polarity'] = jnp.ones_like(jnp_data['max_current_slope'])

        self.stimulator = jnp_data

    def get_or_create_jitted_step(self):
        """
        On first call, creates and JIT-compiles the single-step optimizer function.
        Subsequently, fetches the created function handle.
        """ 
        if not self.jitted_step:
            # Initialize the Differential Evolution strategy.
            dummy_population = jnp.ones(self.n_coils) * (self.MSO_constraint * 0.9)
            self.strategy = DE(
                population_size=self.optimizer_settings['popsize'],
                solution=dummy_population,
            )
            self.strategy_params = self.strategy.default_params.replace(
                crossover_rate=self.optimizer_settings['crossover_rate'],
                differential_weight=self.optimizer_settings['differential_weight'],
            )

            self.jitted_step = jax.jit(
                partial(
                    _optimizer_step,
                    strategy=self.strategy,
                    strategy_params=self.strategy_params,
                    jitted_objective=self.jitted_objective
                )
            )
        return self.jitted_step

    def run(self, target: Tuple) -> jaxArray:
        jitted_step = self.get_or_create_jitted_step()

        settings = self.optimizer_settings
        key = jax.random.key(0)

        # Convert dynamic data (which changes each trial) to JAX arrays.
        position, direction = target
        target_jax = (jax.device_put(np.asarray(position), self.jax_device), 
                      jax.device_put(np.asarray(direction), self.jax_device))

        # Initialize population using a Sobol sequence for quasi-random coverage of the search space.
        initial_population = self.sobol_sampler.random(n=settings['popsize']) * 2 - 1
        initial_population = self.constraint_solution_energy(initial_population).astype(np.float32)
        initial_population = jax.device_put(initial_population, self.jax_device)

        initial_fitness = self.jitted_objective(initial_population, target_jax)

        # Initialize the optimizer state.
        key, subkey = jax.random.split(key)
        state = self.strategy.init(subkey, initial_population, initial_fitness, self.strategy_params)

        # Run the optimization loop efficiently using jax.lax.scan.
        initial_carry = (state, key, target_jax)
        scan_xs = jnp.arange(settings['num_generations'])
        final_carry, metrics_log = jax.lax.scan(jitted_step, initial_carry, scan_xs)

        # Block until computation is finished to get results.
        final_state = final_carry[0]
        final_state.best_solution.block_until_ready()
        metrics_log['best_fitness'].block_until_ready()

        self.losses = np.array(metrics_log['best_fitness'])

        # Convert the best solution back to a PyTorch tensor and ensure it
        # respects the energy constraint.
        best_population = np.array(final_state.best_solution)
        best_population = self.constraint_solution_energy(best_population)

        final_population = best_population * self.stimulator['max_current_slope']

        return final_population


    def constraint_solution_energy(self, x: jaxArray) -> jaxArray:
        """
        Ensures a stimulus solution does not exceed the maximum energy limit using JAX.

        Args:
            x: An array of stimulus solutions, shape (n_solutions, n_coils) or (n_coils,).

        Returns:
            The input array, scaled down if necessary to meet the energy constraint.
        """
        x = jnp.asarray(x, dtype=jnp.float32)
        
        energy_limit = jnp.sum((jnp.ones(self.n_coils) * self.MSO_constraint)**2)
        
        original_ndim = x.ndim
        if original_ndim == 1:
            x = x[jnp.newaxis, :]
            
        solution_energy = jnp.sum(x**2, axis=-1)
        energy_relation = solution_energy / energy_limit
        violation_mask = energy_relation > 1.0

        # Only compute for and apply scaling to violating solutions
        if jnp.any(violation_mask):
            violating_relations = energy_relation[violation_mask]
            scale_factors = jnp.sqrt(violating_relations)[:, jnp.newaxis]
            
            # Update only the violating rows using JAX's functional update syntax
            x = x.at[violation_mask].set((x[violation_mask] / scale_factors) * 0.999)
        
        if original_ndim == 1:
            return x.squeeze(axis=0)
        return x