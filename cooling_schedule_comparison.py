import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns
from matplotlib.colors import PowerNorm
from simulated_annealing import *
from cooling_functions import *

sns.set_style("whitegrid")



cooling_schedules = [
    (exponential_cooling, 0.9999),
    # (linear_cooling, 0.0001),
    (logarithmic_cooling, 0.001),
    # (quadratic_cooling, 0.001),
    # (fast_annealing, 0.001),
    (boltzmann_cooling, 1)
]



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

    fig, ax_energy = plt.subplots(figsize=(10, 6))

    for cooling_function, cooling_parameter in cooling_schedules:
        all_energies = []

        for _ in range(num_runs):
            initial_particles = initial_configuration(num_particles)
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
            normalized_energies = 100 * np.array(energies) / energies[0]
            if len(normalized_energies) < max_display_iterations:
                padding_length = max_display_iterations - len(normalized_energies)
                padded_energies = np.pad(normalized_energies, (0, padding_length), 'constant', constant_values=(normalized_energies[-1],))
            else:
                padded_energies = normalized_energies[:max_display_iterations]

            all_energies.append(padded_energies)

        # Calculate the mean and 95% confidence interval for each cooling schedule
        all_energies = np.array(all_energies)
        mean_energies = np.mean(all_energies, axis=0)
        ci = stats.sem(all_energies, axis=0) * stats.t.ppf((1 + 0.95) / 2, num_runs - 1)

        ax_energy.plot(range(max_display_iterations), mean_energies, label=f"{format_function_name(cooling_function)} (param: {cooling_parameter})")
        ax_energy.fill_between(range(max_display_iterations), mean_energies - ci, mean_energies + ci, alpha=0.2)

    ax_energy.set_title(f"Normalized Mean System Energy with 95% Confidence Interval for {num_particles} Particles")
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("Normalized Energy (%)")
    ax_energy.legend()

    plt.show()

if __name__ == "__main__":
    main()
