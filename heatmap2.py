import numpy as np
import cProfile
from matplotlib.colors import Normalize, PowerNorm, LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import simulation as ps
from multiprocessing import Pool
from numba import jit
import matplotlib.colors as mcolors

from simulation import *
from simulated_annealing import *
from cooling_functions import *


def simulate_and_visualize(num_particles, radius, initial_temp, cooling_function, cooling_parameter, max_step, tolerance, max_consecutive_iterations, boundary_condition, max_energy):
    initial_particles = initial_configuration(num_particles)
    best_particles, particle_history, energies = simulated_annealing(
        initial_particles,
        radius,
        initial_temp,
        cooling_function,
        max_step,
        tolerance,
        max_consecutive_iterations,
        cooling_parameter,
        boundary_condition,
        max_energy
    )
    return best_particles, particle_history, energies


def create_custom_colormap(original_cmap, start=0, stop=1):
    """
    Create a custom colormap using a portion of an existing colormap.

    :param original_cmap: Original colormap (e.g., plt.cm.viridis).
    :param start: Starting point of the new colormap (0 to 1).
    :param stop: Ending point of the new colormap (0 to 1).
    :return: A new colormap.
    """
    colors = original_cmap(np.linspace(start, stop, 256))
    reversed_colors = colors[::-1]
    new_cmap = mcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=original_cmap.name, a=start, b=stop), reversed_colors)
    return new_cmap


@jit
def gather_all_final_positions(num_charges_list, num_simulations, params, max_energy):
    all_positions_data = {}
    for num_charges in num_charges_list:
        all_final_positions = np.zeros((num_simulations * num_charges, 2))
        for i in range(num_simulations):
            final_positions = get_final_positions(num_charges, params, max_energy)
            all_final_positions[i * num_charges:(i + 1) * num_charges] = final_positions
            print(f"Simulation {i + 1} of {num_simulations} complete for {num_charges} charges")
        all_positions_data[num_charges] = all_final_positions
    return all_positions_data




@jit
def get_final_positions(num_charges, params, max_energy):
    _, particle_history, _ = simulate_and_visualize(num_charges, *params, max_energy)
    return particle_history[-1]




def plot_combined_heatmap(all_final_positions, radius=1):
    plt.hist2d(all_final_positions[:, 0], all_final_positions[:, 1], bins=200, range=[[-radius, radius], [-radius, radius]], cmap='hot', density=True)
    plt.title(f"Heatmap of Final Positions for {num_charges} Particles")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label='Density')
    circle = plt.Circle((0, 0), radius, color='blue', fill=False)
    plt.gca().add_artist(circle)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
def plot_radial_distribution(all_final_positions, radius=1):
    radial_distances = np.sqrt(all_final_positions[:, 0]**2 + all_final_positions[:, 1]**2)
    counts, bin_edges = np.histogram(radial_distances, bins=200, range=[0, radius], density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    plt.semilogy(bin_centers, counts, color='blue', alpha=0.7)
    plt.title(f"Radial Distribution of Final Positions for {num_charges} Particles and {num_simulations_per_charge} Simulations")
    plt.xlabel("Radial Distance")
    plt.ylabel("Density (Log Scale)")
    plt.xlim(0, radius)
    plt.show()
    
def plot_radial_distribution_gradient(all_final_positions, num_charges, radius=1):
    radial_distances = np.sqrt(all_final_positions[:, 0]**2 + all_final_positions[:, 1]**2)
    counts, bin_edges = np.histogram(radial_distances, bins=1000, range=[0, radius], density=True)
    norm = LogNorm(vmin=counts[counts > 0].min(), vmax=counts.max())
    original_cmap = plt.cm.viridis
    cmap = create_custom_colormap(original_cmap, start=0, stop=0.8)

    fig, ax = plt.subplots(figsize=(8, 2))
    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i+1]
        rect = patches.Rectangle((left, 0), right-left, 1, color=cmap(norm(counts[i])))
        ax.add_patch(rect)

    ax.set_xlim(0, radius)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    plt.gca().yaxis.grid(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.3)
    cbar.set_label('Density')
    plt.title(f"Radial Distribution Gradient for {num_charges} Particles and {num_simulations_per_charge} Simulations")
    filename = f"radial_distribution_gradient_{num_charges}_particles_{num_simulations_per_charge}_sims.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()





if __name__ == "__main__":
    
    def initial_configuration(num_particles):
        particles = []
        epsilon = np.random.uniform(-0.1, 0.1)
        angle_increment = 2 * np.pi / num_particles
        for i in range(num_particles):
            radius = 0.5 * epsilon
            angle = i * epsilon*angle_increment
            particles.append(radius * np.array([np.cos(angle), np.sin(angle)]))
        return np.array(particles)
    
    # Simulation parameters
    num_charges_list = [11]  # Example list of different numbers of charges
    num_simulations_per_charge = 100
    boundary_condition = "circular"   # "circular" or "periodic"

    # Simulation parameters
    radius = 1
    initial_temp = 10000
    final_temp = 0.001
    max_step = 0.0025
    tolerance = 0.001
    max_consecutive_iterations = 1000

    # Cooling function parameters
    cooling_parameter = 0.9999
    cooling_function = boltzmann_cooling

    params = (radius, initial_temp, cooling_function, cooling_parameter, max_step, tolerance, max_consecutive_iterations, boundary_condition)
    
    for num_charges in num_charges_list:
        max_config_particles = maximum_energy_configuration(num_charges, radius)
        max_energy = calculate_energy(max_config_particles, 1)
        all_final_position_data = gather_all_final_positions([num_charges], num_simulations_per_charge, params, max_energy)[num_charges]
        plot_radial_distribution_gradient(all_final_position_data, num_charges, radius)