import numpy as np
import cProfile
from matplotlib.colors import Normalize, PowerNorm, LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import simulation as ps
from simulation import simulate_and_visualize
from multiprocessing import Pool
from numba import jit
import matplotlib.colors as mcolors


"""
This script runs a series of simulations and plots the final positions of the particles as a heatmap.

"""


# Simulation parameters
num_charges = 50
num_simulations_per_charge = 100 

# Simulation parameters
radius = 1
initial_temp = 10000
final_temp = 0.001
cooling_rate = 0.9999
max_step = 0.01
tolerance = 0.001
max_consecutive_iterations = 10 

params = (radius, initial_temp, final_temp, cooling_rate, max_step, tolerance, max_consecutive_iterations)


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
def gather_all_final_positions(num_charges, num_simulations, params):
    all_final_positions = np.zeros((num_simulations * num_charges, 2)) 
    for i in range(num_simulations):
        final_positions = get_final_positions(num_charges, params)
        all_final_positions[i * num_charges:(i + 1) * num_charges] = final_positions
        print(f"Simulation {i + 1} of {num_simulations} complete")
    return all_final_positions

@jit
def get_final_positions(num_charges, params):
    _, particle_history, _ = simulate_and_visualize(num_charges, *params)
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
    
def plot_radial_distribution_gradient(all_final_positions, radius=1):
    radial_distances = np.sqrt(all_final_positions[:, 0]**2 + all_final_positions[:, 1]**2)
    counts, bin_edges = np.histogram(radial_distances, bins=1000, range=[0, radius], density=True)

    # Normalize the counts for color mapping
    norm = LogNorm(vmin=counts[counts > 0].min(), vmax=counts.max())  # Avoid log(0) issues

    # Creating the colormap
    original_cmap = plt.cm.viridis
    cmap = create_custom_colormap(original_cmap, start=0, stop=0.8)
    
    fig, ax = plt.subplots(figsize=(8, 2))

    # Create a series of rectangles to form the gradient bar
    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i+1]
        rect = patches.Rectangle((left, 0), right-left, 1, color=cmap(norm(counts[i])))
        ax.add_patch(rect)

    ax.set_xlim(0, radius)
    ax.set_ylim(0, 1)
    # ax.set_axis_off()
    ax.set_yticklabels([])
    plt.gca().yaxis.grid(False)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.1)
    cbar.set_label('Density')

    plt.title(f"Radial Distribution Gradient for {num_charges} Particles and {num_simulations_per_charge} Simulations")
    plt.show()






if __name__== "__main__":
    # cProfile.run('gather_all_final_positions(num_charges, num_simulations_per_charge, params)', 'profile_stats.prof')
    all_final_position_data = gather_all_final_positions(num_charges, num_simulations_per_charge, params)

    # plot_combined_heatmap(all_final_position_data, radius)
    plot_radial_distribution_gradient(all_final_position_data, radius)
    


