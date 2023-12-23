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
from initial_configs import *

plt.rcParams.update({'font.size': 15})


class CustomNormalize(Normalize):
    def __init__(self, vcenter, vmax, clip=False):
        self.vcenter = vcenter
        Normalize.__init__(self, vmin=None, vmax=vmax, clip=clip)

    def __call__(self, value, clip=None):
        x, y = [self.vcenter, self.vmax], [0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def simulate_and_visualize(num_particles, radius, initial_temp, cooling_function, cooling_parameter, max_step, tolerance, max_consecutive_iterations, boundary_condition, max_energy, initial_config_func):
    initial_particles = initial_config_func(num_particles)
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
def gather_all_final_positions(num_charges_list, num_simulations, params, max_energy, initial_config_func):
    all_positions_data = {}
    for num_charges in num_charges_list:
        all_final_positions = np.zeros((num_simulations * num_charges, 2))
        for i in range(num_simulations):
            final_positions = get_final_positions(num_charges, params, max_energy, initial_config_func)
            all_final_positions[i * num_charges:(i + 1) * num_charges] = final_positions
            print(f"Simulation {i + 1} of {num_simulations} complete for {num_charges} charges")
        all_positions_data[num_charges] = all_final_positions
    return all_positions_data




@jit
def get_final_positions(num_charges, params, max_energy, initial_config_func):
    _, particle_history, _ = simulate_and_visualize(num_charges, *params, max_energy, initial_config_func)
    return particle_history[-1]



    
def plot_combined_radial_distribution_gradients(all_positions_data, num_charges_list, radius):
    fig, ax = plt.subplots(figsize=(8, len(num_charges_list) * 2))
    for idx, num_charges in enumerate(num_charges_list):
        all_final_positions = all_positions_data[num_charges]
        radial_distances = np.sqrt(all_final_positions[:, 0]**2 + all_final_positions[:, 1]**2)
        
        # Check if all values are at 1
        if np.all(radial_distances == 1):
            # Plot a vertical line at radial distance 1 for this number of charges
            ax.vlines(1, idx, idx + 1, colors='purple', linewidth=1)
        else:
            # Execute the original gradient plot code
            counts, bin_edges = np.histogram(radial_distances, bins=500, range=[0, radius], density=True)
            norm = LogNorm(vmin=counts[counts > 0].min(), vmax=counts.max())
            cmap = create_custom_colormap(plt.cm.viridis, start=0, stop=0.93)
            
            for i in range(len(bin_edges) - 1):
                left, right = bin_edges[i], bin_edges[i+1]
                rect = patches.Rectangle((left, idx), right-left, 1, color=cmap(norm(counts[i])))
                ax.add_patch(rect)
    
    ax.set_xlim(-0.1, 1.1*radius)
    ax.set_ylim(0, len(num_charges_list))
    ax.set_yticks(np.arange(len(num_charges_list)) + 0.5)
    ax.set_yticklabels([f"{nc}" for nc in num_charges_list])
    ax.set_xlabel('Radial Distance')
    ax.set_ylabel('Number of Charges')
    
    # plt.title(f"Combined Radial Distribution Gradient")
    plt.savefig(f"combined_rdg_{initial_temp}_{max_step}_{cooling_parameter}_{num_charges_list}_{num_simulations_per_charge}.png", dpi=300, bbox_inches='tight')    
    plt.show()

def choose_initial_config_function(num_charges):
    if num_charges <= 11:
        return initial_configuration_11
    elif 12 <= num_charges <= 16:
        return initial_configuration_12
    elif 17 <= num_charges <= 29:
        return initial_configuration_17_to_29
    elif 30 <= num_charges <= 37:
        return initial_configuration_30_to_37
    else:
        raise ValueError("No configuration function for the given number of charges")




if __name__ == "__main__":
    
    # Simulation parameters
    num_simulations_per_charge = 50
    boundary_condition = "circular"   # "circular" or "periodic"

    # Simulation parameters
    radius = 1
    initial_temp = 1000
    final_temp = 0.001
    max_step = 0.1
    tolerance = 0.001
    max_consecutive_iterations = 250

    # Cooling function parameters
    cooling_parameter = 0.001
    cooling_function = quadratic_cooling

    params = (radius, initial_temp, cooling_function, cooling_parameter, max_step, tolerance, max_consecutive_iterations, boundary_condition)
    
    # num_charges_list = list(range(10, 12))
    # num_charges_list = list(range(10, 14))
    num_charges_list = list(range(15, 19))
    # num_charges_list = list(range(28, 32))    
    all_data = {}


    for num_charges in num_charges_list:
        initial_config_func = choose_initial_config_function(num_charges)
        max_config_particles = maximum_energy_configuration(num_charges, radius)
        max_energy = calculate_energy(max_config_particles, 1)
        all_final_position_data = gather_all_final_positions([num_charges], num_simulations_per_charge, params, max_energy, initial_config_func)[num_charges]
        all_data[num_charges] = all_final_position_data
        


    plot_combined_radial_distribution_gradients(all_data, num_charges_list, radius)