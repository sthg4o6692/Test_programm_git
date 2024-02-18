import numpy as np
import matplotlib.pyplot as plt

def glauber_algorithm(temperature, lattice_size, num_iterations):
    lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
    magnetization = []
    
    for _ in range(num_iterations):
        # Randomly select a spin from the lattice
        i, j = np.random.randint(0, lattice_size, size=2)
        
        # Calculate the change in energy (∆E) if the selected spin were to flip
        delta_E = 2 * lattice[i, j] * (lattice[(i+1)%lattice_size, j] + lattice[(i-1)%lattice_size, j] +
                                      lattice[i, (j+1)%lattice_size] + lattice[i, (j-1)%lattice_size])
        
        # Calculate the transition probability P for flipping the spin
        p = 1 / (1 + np.exp(temperature * delta_E))
        
        # Generate a random number r between 0 and 1
        r = np.random.rand()
        
        # If r < P, flip the selected spin. Otherwise, leave the spin unchanged
        if r < p:
            lattice[i, j] *= -1
        
        # Calculate the magnetization of the lattice
        magnetization.append(np.sum(lattice) / (lattice_size ** 2))
    
    return magnetization

def metropolis_algorithm(temperature, lattice_size, num_iterations):
    lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
    magnetization = []
    
    for _ in range(num_iterations):
        # Randomly select a spin from the lattice
        i, j = np.random.randint(0, lattice_size, size=2)
        
        # Calculate the change in energy (∆E) if the selected spin were to flip
        delta_E = 2 * lattice[i, j] * (lattice[(i+1)%lattice_size, j] + lattice[(i-1)%lattice_size, j] +
                                      lattice[i, (j+1)%lattice_size] + lattice[i, (j-1)%lattice_size])
        
        # If ∆E <= 0, flip the selected spin. Otherwise, flip the spin with a probability of exp(-∆E / temperature)
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / temperature):
            lattice[i, j] *= -1
        
        # Calculate the magnetization of the lattice
        magnetization.append(np.sum(lattice) / (lattice_size ** 2))
    
    return magnetization

# Parameters
temperature = 1.5  # Temperature (T < Tc)
lattice_size = 64  # Size of the lattice
num_iterations = 10000  # Number of iterations

# Run Glauber algorithm
glauber_magnetization = glauber_algorithm(temperature, lattice_size, num_iterations)

# Run Metropolis algorithm
metropolis_magnetization = metropolis_algorithm(temperature, lattice_size, num_iterations)

# Plot magnetization vs Monte Carlo steps
plt.plot(range(num_iterations), glauber_magnetization, label='Glauber')
plt.plot(range(num_iterations), metropolis_magnetization, label='Metropolis')
plt.xlabel('Monte Carlo steps')
plt.ylabel('Magnetization')
plt.legend()
plt.show()

# Parameters
lattice_size = 64  # Size of the lattice
num_iterations = 10000  # Number of iterations

# Temperature range
temperatures = np.arange(1.0, 5.1, 0.1)

# Initialize lists to store equilibrium magnetization and errors
glauber_equilibrium_magnetization = []
metropolis_equilibrium_magnetization = []
glauber_errors = []
metropolis_errors = []

# Run simulations for each temperature
for temperature in temperatures:
    # Run Glauber algorithm
    glauber_magnetization = glauber_algorithm(temperature, lattice_size, num_iterations)
    glauber_equilibrium_magnetization.append(np.mean(glauber_magnetization[-1000:]))  # Average of last 1000 magnetization values
    glauber_errors.append(np.std(glauber_magnetization[-1000:]))  # Standard deviation of last 1000 magnetization values
    
    # Run Metropolis algorithm
    metropolis_magnetization = metropolis_algorithm(temperature, lattice_size, num_iterations)
    metropolis_equilibrium_magnetization.append(np.mean(metropolis_magnetization[-1000:]))  # Average of last 1000 magnetization values
    metropolis_errors.append(np.std(metropolis_magnetization[-1000:]))  # Standard deviation of last 1000 magnetization values

# Plot equilibrium magnetization vs. temperature
plt.errorbar(temperatures, glauber_equilibrium_magnetization, yerr=glauber_errors, label='Glauber')
plt.errorbar(temperatures, metropolis_equilibrium_magnetization, yerr=metropolis_errors, label='Metropolis')
plt.xlabel('Temperature')
plt.ylabel('Equilibrium Magnetization')
plt.legend()
plt.show()
