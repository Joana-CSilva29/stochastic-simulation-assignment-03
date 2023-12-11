import numpy as np
import matplotlib.pyplot as plt

def potential_energy_circle(n):
    omega_n = (n/4)*sum(1 / np.sin((np.pi * j) / n) for j in range(1, n))
    return omega_n

def potential_energy_1_charge_in_middle(n):
    n=n-1
    omega_n_circle = potential_energy_circle(n) + (n)
    return omega_n_circle




n_values = list(range(3, 25))
omega_circle = [potential_energy_circle(n) for n in n_values]
omega_center = [potential_energy_1_charge_in_middle(n) for n in n_values]


for n in n_values:
    print(n, potential_energy_circle(n), potential_energy_1_charge_in_middle(n))

plt.figure(figsize=(10, 6))

crossover_index = np.argmax(np.array(omega_circle) > np.array(omega_center))
crossover_n = n_values[crossover_index]

plt.plot(n_values[:crossover_index+1], omega_circle[:crossover_index+1], label='All charges on the circle', marker='o', color='blue')
plt.plot(n_values[:crossover_index+1], omega_center[:crossover_index+1], label='One charge in the center', marker='x', color='orange')

plt.plot(n_values[crossover_index:], omega_circle[crossover_index:], marker='o', linestyle='--', color='blue')
plt.plot(n_values[crossover_index:], omega_center[crossover_index:], marker='x', linestyle='--', color='orange')


plt.annotate('Crossover Point (n={})'.format(crossover_n),
             xy=(crossover_n, omega_circle[crossover_index]), 
             xytext=(crossover_n+4, omega_circle[crossover_index]),
             ha='center')

plt.xlabel('Number of charges (n)')
plt.ylabel(r'Potential Energy ($\Omega$)')
plt.title('Potential Energy Configurations')
plt.legend()
plt.grid(True)
plt.show()
