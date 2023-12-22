import math

"""
This module contains the cooling functions used in simulated annealing.

"""


def exponential_cooling(initial_temp, cooling_rate, iteration):
    return initial_temp * (cooling_rate ** iteration)

def linear_cooling(initial_temp, cooling_rate, iteration):
    return max(initial_temp - cooling_rate * iteration, 0)

def boltzmann_cooling(initial_temp, iteration, _unused):
    a=1
    b=1
    if iteration == 0:
        return initial_temp
    return (a * initial_temp) / math.log(b + iteration)

def logarithmic_cooling(initial_temp, cooling_rate, iteration):
    a=1
    b=1
    return initial_temp / (1 + a * cooling_rate * math.log(b + iteration))

def quadratic_cooling(initial_temp, cooling_rate, iteration):
    return initial_temp / (1 + cooling_rate * (iteration ** 2))

def fast_annealing(initial_temp, cooling_rate, iteration):
    return initial_temp / (1 + cooling_rate * iteration)