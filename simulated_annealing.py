import numpy as np
from numba import jit

"""
    This module contains the simulated annealing algorithm and utility functions.
"""


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


def potential_energy_circle(n):
    omega_n = (n / 4) * sum(1 / np.sin((np.pi * j) / n) for j in range(1, n))
    return omega_n


# Calculate energy
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






# Simulated annealing
def simulated_annealing(particles, radius, initial_temp, cooling_function, max_step, tolerance, max_consecutive_iterations, cooling_parameter, boundary_condition, max_energy):
    temperature = initial_temp
    iteration = 0
    best_energy = calculate_energy(particles, max_energy)
    best_particles = np.copy(particles)
    particle_history = [np.copy(particles)]
    energies = [best_energy]

    consecutive_low_change_count = 0

    while temperature > 0:
        new_particles = np.copy(particles)
        for i in range(len(particles)):
            # Update temperature for each iteration
            temperature = cooling_function(initial_temp, cooling_parameter, iteration)
            new_temp_ratio = temperature / initial_temp
            new_particle = move_particle(particles[i], particles, max_step * new_temp_ratio, radius, boundary_condition)
            new_particles[i] = new_particle

        new_energy = calculate_energy(new_particles, max_energy)
        energy_change = new_energy - calculate_energy(particles, max_energy)

        # Metropolis criterion
        if energy_change < 0 or np.random.uniform() < np.exp(-energy_change / temperature):
            particles = new_particles
            current_energy = new_energy
        else:
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
        particle_history.append(np.copy(particles))
        energies.append(current_energy)

    return best_particles, particle_history, energies

