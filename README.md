# Stochastic Simulation Assignment 03 - Finding The Minimal Energy Configuration Of Charge Particle Within A Circle

### License:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains Python scripts developed for the third assignment of Stochastic Simulation at the University of Amsterdam, 2023. The project focuses on optimisation of repellent point charges in a circle using Simulated Annealing.

## File Descriptions

### `simulated_annealing.py`
This module includes the core methods used in this project. It contains the implementation of the basic simulated annealing function, as well as the functions to move particles and calculate the energy.

### `cooling_functions.py`
`cooling_functions.py` contains the cooling functions that are used during the simulated annealing process.

### `simulation.py`
The main animation and visualisation script. Different parameters, such as step sizes, initial temperature and cooling parameters can be set for any number of particles.

### `heatmap.py`
This script runs the simulation multiple times and creates a heatmap of the final particle positions, which can be used to find the magic numbers.

### `cooling_comparison.py`
Here we provide the functionality to compare the performance of different cooling schedules across a variety of measures.

###  `statistical_test.py`
Basic tests to evaluate our data statistically.

### `analytical_solution.py`
In order to find the first magic number, we calculated the energy of the system analytically.

### `profiling.py`
This is able to analyse the .prof file the main scripts spit out to optimise performance.

## Usage
These scripts were run with Python 3.11.0 on MacOS Ventura. 
The main functions were converted to machine code using @jit in order to speed up the simulation. This was disabled for the animations.

### Requirements:
matplotlib==3.7.1
numba==0.57.0
numpy==1.24.3
scipy==1.10.1
seaborn==0.13.0


## Contact
joana.costaesilva@student.uva.nl
balint.szarvas@student.uva.nl
sandor.battaglini-fischer@student.uva.nl

---

Developed by Joana Costa e Silva, Bálint Szarvas and Sándor Battaglini-Fischer.
