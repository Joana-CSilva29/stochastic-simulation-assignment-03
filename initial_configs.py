import numpy as np

epsilon = np.random.uniform(-0.15, 0.15)

def initial_configuration_11(num_particles):
    particles = []
    for i in range(num_particles):
        angle = i * (2 * np.pi / num_particles)
        radius = 0.9
        particles.append(radius * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(particles)

def initial_configuration_12(num_particles):
    particles = []
    if num_particles > 0:
        particles.append(np.array([0, 0]))

    for i in range(1, num_particles):
        angle = i * (2 * np.pi / (num_particles - 1)) + np.random.uniform(-5*epsilon, 5*epsilon)
        radius = np.random.uniform(0.85, 0.95)
        particles.append(radius * np.array([np.cos(angle), np.sin(angle)]))
    
    return np.array(particles)

def initial_configuration_17_to_29(num_particles):
    particles = []
    num_inner_particles = np.random.randint(2, 10)
    for _ in range(num_inner_particles):
        particles.append([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)])
    for _ in range(num_particles - num_inner_particles):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.49, 0.5)
        particles.append(radius * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(particles)

def initial_configuration_30_to_37(num_particles):
    if num_particles < 30 or num_particles > 37:
        raise ValueError("Number of particles must be between 30 and 37")

    particles = []
    if num_particles > 1:
        particles.append([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)])
    

    inner_ring_particles = np.random.randint(2, 10) 

    inner_radius = np.random.uniform(0.30, 0.35) 
    outer_radius = np.random.uniform(0.55, 0.55)

    for i in range(inner_ring_particles):
        angle = i * (2 * np.pi / inner_ring_particles)
        particles.append(inner_radius * np.array([np.cos(angle), np.sin(angle)]))

    outer_ring_particles = num_particles - 1 - inner_ring_particles
    for i in range(outer_ring_particles):
        angle = i * (2 * np.pi / outer_ring_particles)
        particles.append(outer_radius * np.array([np.cos(angle), np.sin(angle)]))

    return np.array(particles)