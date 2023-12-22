import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns
from matplotlib.colors import PowerNorm

sns.set_style("whitegrid")

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

cooling_schedules = [
    (exponential_cooling, 0.9999),
    (linear_cooling, 0.0001),
    (logarithmic_cooling, 0.001),
    (quadratic_cooling, 0.001),
    (fast_annealing, 0.001),
    (boltzmann_cooling, 1)
]

def potential_energy_circle(n):
    omega_n = (n / 4) * sum(1 / np.sin((np.pi * j) / n) for j in range(1, n))
    return omega_n

def calculate_energy(particles, reference_energy):
    energy = 0
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            distance = np.linalg.norm(particles[i] - particles[j])
            energy += 1 / distance
    return energy / reference_energy

def initial_configuration(num_particles):
    particles = []
    for i in range(num_particles):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, 0.1)
        particles.append(radius * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(particles)

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

    return new_particle

def simulated_annealing(particles, radius, initial_temp, cooling_function, max_step, tolerance, max_consecutive_iterations, cooling_parameter, boundary_condition, reference_energy):
    temperature = initial_temp
    iteration = 0
    best_energy = calculate_energy(particles, reference_energy)
    best_particles = np.copy(particles)
    particle_history = [np.copy(particles)]
    energies = [best_energy]

    consecutive_low_change_count = 0

    while temperature > 0:
        for i in range(len(particles)):
            temperature = cooling_function(initial_temp, cooling_parameter, iteration)
            new_temp_ratio = temperature / initial_temp
            particles[i] = move_particle(particles[i], particles, max_step * new_temp_ratio, radius, boundary_condition)

        current_energy = calculate_energy(particles, reference_energy)
        best_energy = calculate_energy(best_particles, reference_energy)

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

def simulate_and_visualize(initial_particles, radius, initial_temp, cooling_function, max_step, tolerance, max_consecutive_iterations, cooling_parameter, boundary_condition, reference_energy):
    particles = np.copy(initial_particles)
    best_particles, particle_history, energies = simulated_annealing(
        particles, 
        radius, 
        initial_temp, 
        cooling_function, 
        max_step, 
        tolerance, 
        max_consecutive_iterations, 
        cooling_parameter,
        boundary_condition,
        reference_energy
    )

    return best_particles, particle_history, energies


def run_multiple_simulations(initial_particles, num_runs, radius, initial_temp, cooling_function, max_step, tolerance, max_consecutive_iterations, cooling_parameter, boundary_condition, reference_energy, max_display_iterations):
    all_energies = []

    for _ in range(num_runs):
        _, _, energies = simulate_and_visualize(
            initial_particles, 
            radius, 
            initial_temp, 
            cooling_function, 
            max_step, 
            tolerance, 
            max_consecutive_iterations, 
            cooling_parameter,
            boundary_condition,
            reference_energy
        )
        energies = energies[:max_display_iterations] + [energies[-1]] * (max_display_iterations - len(energies))
        all_energies.append(energies)

    return np.array(all_energies)

def main():
    num_runs = 30
    boundary_condition = "circular"  # "circular" or "periodic"

    num_particles = 12
    radius = 1
    initial_temp = 10
    max_step = 0.02
    tolerance = 0.001
    max_consecutive_iterations = 100
    max_display_iterations = 50

    reference_energy = potential_energy_circle(num_particles)
    initial_particles = initial_configuration(num_particles)

    fig, ax_energy = plt.subplots(figsize=(10, 6))

    for cooling_function, cooling_parameter in cooling_schedules:
        all_energies = run_multiple_simulations(
            initial_particles, 
            num_runs, 
            radius, 
            initial_temp, 
            cooling_function, 
            max_step, 
            tolerance, 
            max_consecutive_iterations, 
            cooling_parameter,
            boundary_condition,
            reference_energy,
            max_display_iterations
        )

        mean_energies = np.mean(all_energies, axis=0)
        ci = stats.sem(all_energies, axis=0) * stats.t.ppf((1 + 0.95) / 2., num_runs - 1)  # 95% confidence interval

        ax_energy.plot(range(max_display_iterations), mean_energies, label=f"{cooling_function.__name__} (param: {cooling_parameter})")
        ax_energy.fill_between(range(max_display_iterations), mean_energies - ci, mean_energies + ci, alpha=0.2)

    ax_energy.set_title(f"Mean System Energy with 95% Confidence Interval for {num_particles} Particles")
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("Energy")
    ax_energy.legend()

    plt.show()

if __name__ == "__main__":
    main()