import numpy as np
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.cm import ScalarMappable
from numba import jit
import cProfile
from cooling_functions import *
from simulated_annealing import *

sns.set_style("whitegrid")


# Initial configuration
def initial_configuration(num_particles):
    particles = []
    for i in range(num_particles):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.99, 1)
        particles.append(radius * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(particles)


# Update plot function for animation
def update_plot(frame, particles, scat, radius, energies, cmap, norm, time_text, table, energy_line, ax_energy, display_table):
    particle_positions = particles[frame]
    scat.set_offsets(particle_positions)

    current_energy = energies[frame]
    norm = PowerNorm(gamma=0.1, vmin=min(energies), vmax=max(energies))
    colors = cmap(norm(current_energy))
    scat.set_color(colors)

    time_text.set_text(f"Step: {frame}\nParticles: {len(particle_positions)}")

    if display_table and table is not None:
        polar_coords = [cartesian_to_polar(x, y) for x, y in particle_positions]
        table_data = [["Particle", "Radius", "Phi"]] 
        for i, (r, phi) in enumerate(polar_coords):
            table_data.append([f"{i+1}", f"{r:.2f}", f"{phi:.0f}°"])

        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                table._cells[(i, j)].get_text().set_text(cell)

    energy_line.set_data(range(frame+1), energies[:frame+1])
    ax_energy.set_xlim(0, frame+1)
    ax_energy.relim()
    ax_energy.autoscale_view()

    return scat, time_text

# Simulate and visualize the animation
def simulate_and_visualize(num_particles, radius, initial_temp, cooling_function, max_step, tolerance, max_consecutive_iterations, cooling_parameter, boundary_condition, max_energy):
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



def main():
    boundary_condition = "periodic" # "circular" or "periodic"
    num_particles = 12
    radius = 1
    initial_temp = 10
    max_step = 0.02
    tolerance = 0.001
    max_consecutive_iterations = 1000 

    table = None
    display_table = False

    cooling_parameter = 0.9999  
    cooling_function = exponential_cooling
    
    max_config_particles = maximum_energy_configuration(num_particles, radius)
    max_energy = calculate_energy(max_config_particles, 1)

    best_particles, particle_history, energies = simulate_and_visualize(num_particles, radius, initial_temp, cooling_function, max_step, tolerance, max_consecutive_iterations, cooling_parameter, boundary_condition, max_energy)

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 2, 1])
    ax = fig.add_subplot(gs[0])
    ax_energy = fig.add_subplot(gs[1])
    ax_table = fig.add_subplot(gs[2])
    ax_table.axis('tight')
    ax_table.axis('off')

    circle = plt.Circle((0, 0), radius, color='r', fill=False)
    center = plt.Circle((0, 0), 0.01, color='r', fill=True)
    ax.add_artist(circle)
    ax.add_artist(center)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal', adjustable='box')

    scat = ax.scatter(particle_history[0][:, 0], particle_history[0][:, 1])
    cmap = plt.colormaps['coolwarm']
    norm = Normalize(vmin=min(energies), vmax=max(energies))
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax_energy.set_title("System Energy for {} Particles".format(num_particles))
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("Energy")
    energy_line, = ax_energy.plot([], [], lw=2)
    ax_energy.set_xlim(0, len(particle_history))
    ax_energy.set_ylim(min(energies), max(energies))

    ani = FuncAnimation(fig, update_plot, frames=len(particle_history), fargs=(particle_history, scat, radius, energies, cmap, norm, time_text, table, energy_line, ax_energy, display_table), interval=10, repeat=False)

    plt.show()



if __name__ == "__main__":    
    cProfile.run('main()', 'profile_stats.prof')