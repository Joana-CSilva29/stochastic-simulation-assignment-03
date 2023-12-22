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

sns.set_style("whitegrid")


# Utility function to convert cartesian to polar coordinates
def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.degrees(np.arctan2(y, x)) % 360
    return r, phi


# Cooling functions
def exponential_cooling(initial_temp, cooling_rate, iteration):
    return initial_temp * (cooling_rate ** iteration)

def linear_cooling(initial_temp, cooling_rate, iteration):
    return max(initial_temp - cooling_rate * iteration, 0)

def boltzmann_cooling(initial_temp, iteration, _unused):
    if iteration == 0:
        return initial_temp
    return initial_temp / math.log(1 + iteration)

def logarithmic_cooling(initial_temp, cooling_rate, iteration):
    return initial_temp / (1 + cooling_rate * math.log(1 + iteration))

def quadratic_cooling(initial_temp, cooling_rate, iteration):
    return initial_temp / (1 + cooling_rate * (iteration ** 2))

def fast_annealing(initial_temp, cooling_rate, iteration):
    return initial_temp / (1 + cooling_rate * iteration)


# Maximum energy configuration
def maximum_energy_configuration(num_particles, radius):
    particles = []
    for i in range(num_particles):
        angle = 2 * np.pi * i / num_particles
        particles.append(radius * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(particles)


# Calculate energy
def calculate_energy(particles, max_energy):
    energy = 0
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            distance = np.linalg.norm(particles[i] - particles[j])
            energy += 1 / distance
    return energy / max_energy


# Initial configuration
def initial_configuration(num_particles):
    particles = []
    for i in range(num_particles):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, 0.1)
        particles.append(radius * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(particles)


# Move particle
def move_particle(particle, particles, max_step, radius, boundary_condition):
    force_direction = np.array([0.0, 0.0])
    for other_particle in particles:
        if np.array_equal(particle, other_particle):
            continue
        difference = particle - other_particle
        distance = np.linalg.norm(difference)
        if distance == 0:
            continue
        force_direction += difference / distance**3

    if np.linalg.norm(force_direction) > 0:
        force_direction_normalized = force_direction / np.linalg.norm(force_direction)
        new_particle = particle + force_direction_normalized * max_step
    else:
        new_particle = particle

    if boundary_condition == "circular":
        if np.linalg.norm(new_particle) > radius:
            new_particle = new_particle / np.linalg.norm(new_particle) * radius
    elif boundary_condition == "periodic":
        angle = np.arctan2(new_particle[1], new_particle[0])
        if np.linalg.norm(new_particle) > radius:
            print("Particle escaped")
            new_particle = np.array([np.cos(angle), np.sin(angle)]) * radius * -1

    return new_particle


# Simulated annealing
def simulated_annealing(particles, radius, initial_temp, cooling_function, cooling_parameter, max_step, tolerance, max_consecutive_iterations, boundary_condition, max_energy):
    temperature = initial_temp
    iteration = 0
    best_energy = calculate_energy(particles, max_energy)
    best_particles = np.copy(particles)
    energies = [best_energy]
    consecutive_low_change_count = 0

    while temperature > 0:
        for i in range(len(particles)):
            if cooling_function in [boltzmann_cooling, logarithmic_cooling]:
                temperature = cooling_function(initial_temp, iteration, cooling_parameter)
            else:
                temperature = cooling_function(initial_temp, cooling_parameter, iteration)

            particles[i] = move_particle(particles[i], particles, max_step, radius, boundary_condition)

        current_energy = calculate_energy(particles, max_energy)
        if current_energy < best_energy:
            best_energy = current_energy
            best_particles = np.copy(particles)

        energy_change = abs(energies[-1] - current_energy)
        if energy_change < tolerance:
            consecutive_low_change_count += 1
        else:
            consecutive_low_change_count = 0

        if consecutive_low_change_count >= max_consecutive_iterations:
            break

        iteration += 1
        energies.append(current_energy)

    return best_particles, energies


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


def simulate_and_visualize(num_particles, radius, initial_temp, cooling_function, max_step, tolerance, max_consecutive_iterations, cooling_parameter, boundary_condition, max_energy):
    initial_particles = initial_configuration(num_particles)
    best_particles, particle_history, energies = simulated_annealing(initial_particles, radius, initial_temp, cooling_function, cooling_parameter, max_step, tolerance, max_consecutive_iterations, boundary_condition, max_energy)
    return best_particles, particle_history, energies


# Main function
def main():
    boundary_condition = "circular" # "circular" or "periodic"
    num_particles = 3
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

    plt.show



if __name__ == "__main__":    
    cProfile.run('main()', 'profile_stats.prof')