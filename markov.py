import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from numba import jit
import matplotlib.animation as animation
plt.ion()

# Utilities
def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.degrees(np.arctan2(y, x)) % 360
    return r, phi

def format_function_name(function):
    name = function.__name__
    words = name.split('_')
    title_case_words = [word.capitalize() for word in words]
    return ' '.join(title_case_words)

# Initial energy configuration
def maximum_energy_configuration(num_particles, radius):
    particles = []
    for i in range(num_particles):
        angle = 2 * np.pi * i / num_particles
        particles.append(radius * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(particles)

def calculate_energy(particles, max_energy):
    energy = 0
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            distance = np.linalg.norm(particles[i] - particles[j])
            energy += 1 / distance
    return energy / max_energy

@jit
def move_particle(particle, particles, max_step, radius, boundary_condition):
    force_direction = np.array([0.0, 0.0])
    for other_particle in particles:
        if np.array_equal(particle, other_particle):
            continue

        difference = particle - other_particle
        if boundary_condition == "periodic":
            for i in range(2):
                if abs(difference[i]) > radius:
                    difference[i] -= np.sign(difference[i]) * 2 * radius

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
        if np.linalg.norm(new_particle) > radius:
            angle = np.arctan2(new_particle[1], new_particle[0])
            new_particle = np.array([np.cos(angle), np.sin(angle)]) * radius

    return new_particle


def simulated_annealing(particles, radius, initial_temp, cooling_function, max_step, tolerance, markov_chain_length, cooling_parameter, boundary_condition, max_energy):
    temperature = initial_temp
    iteration = 0
    best_energy = calculate_energy(particles, max_energy)
    best_particles = np.copy(particles)
    particle_history = [np.copy(particles)]
    energies = [best_energy]

    while temperature > 0:
        print(f"Temperature: {temperature}, Iteration: {iteration}")  # Debugging print
        for _ in range(markov_chain_length):
            new_particles = np.copy(particles)
            for i in range(len(particles)):
                new_particle = move_particle(particles[i], particles, max_step, radius, boundary_condition)
                new_particles[i] = new_particle

            new_energy = calculate_energy(new_particles, max_energy)
            energy_change = new_energy - calculate_energy(particles, max_energy)

            if energy_change < 0 or np.random.uniform() < np.exp(-energy_change / temperature):
                particles = new_particles
                current_energy = new_energy
            else:
                current_energy = calculate_energy(particles, max_energy)

            if current_energy < best_energy:
                best_energy = current_energy
                best_particles = np.copy(particles)

            particle_history.append(np.copy(particles))
            energies.append(current_energy)

        iteration += markov_chain_length
        temperature = cooling_function(initial_temp, cooling_parameter, iteration)
    print("Simulation completed for one Markov chain length.")  # Debugging print
    return best_particles, particle_history, energies


# Cooling function
def exponential_cooling(initial_temp, cooling_parameter, iteration):
    return initial_temp * (cooling_parameter ** iteration)

# Main simulation function
def simulate_and_visualize(markov_chain_lengths, num_particles, radius, initial_temp, max_step, tolerance, cooling_parameter, boundary_condition, max_energy):
    results = {}
    for length in markov_chain_lengths:
        print(f"Running simulation for Markov chain length: {length}")
        best_particles, particle_history, energies = simulated_annealing(
            initial_configuration(num_particles),
            radius,
            initial_temp,
            exponential_cooling,
            max_step,
            tolerance,
            length,
            cooling_parameter,
            boundary_condition,
            max_energy
        )
        results[length] = (best_particles, particle_history, energies)
    return results

# Initial configuration
def initial_configuration(num_particles):
    particles = []
    for i in range(num_particles):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.01, 0.02)
        particles.append(radius * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(particles)

def visualize_results(results, radius):
    num_frames = max(len(hist) for _, hist, _ in results.values())

    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))

    scats = []
    for ax in axes:
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect('equal', adjustable='box')
        circle = plt.Circle((0, 0), radius, color='r', fill=False)
        ax.add_artist(circle)
        scat, = ax.plot([], [], 'bo')
        scats.append(scat)

    def init():
        for scat in scats:
            scat.set_data([], [])
        return scats

    def update(frame):
        for idx, (length, (_, particle_history, _)) in enumerate(results.items()):
            if frame < len(particle_history):
                scats[idx].set_data(particle_history[frame][:, 0], particle_history[frame][:, 1])
        return scats

    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=50)

    for idx, length in enumerate(results.keys()):
        axes[idx].set_title(f"Markov Chain Length: {length}")

    plt.show()

# Parameters
num_particles = 12
radius = 1
initial_temp = 10
max_step = 0.02
tolerance = 0.001
cooling_parameter = 0.9999
boundary_condition = "circular"
max_energy = calculate_energy(maximum_energy_configuration(num_particles, radius), 1)

# Markov Chain Lengths to Test
markov_chain_lengths = [1]

# Run Simulation
results = simulate_and_visualize(markov_chain_lengths, num_particles, radius, initial_temp, max_step, tolerance, cooling_parameter, boundary_condition, max_energy)

# Visualize Results
visualize_results(results, radius)
