## Computing the stationary distribution of a Markov chain

import numpy as np

# Compute stationary distribution
# Args: Markov Chain matrix P
def compute_stationary_distribution(transition_matrix):
    # Ensure the matrix is a numpy array
    P = np.array(transition_matrix)

    # We want to solve (P.T - I)x = 0, where P.T is the transpose of the transition matrix
    num_states = P.shape[0]
    
    # np.eye creates an identity matrix of size num_states
    A = P.T - np.eye(num_states)

    # Append the constraint that the sum of probabilities equals 1
    # vstack stacks the arrays vertically (row wise)
    A = np.vstack([A, np.ones(num_states)])
    
    # np.zeros returns a 1D matrix (size of num_states) filled with zeros
    b = np.zeros(num_states)
    
    # append 1 so b = [num_states * 0, 1] to represent the final normalization constraint
    b = np.append(b, 1)

    # b = the right hand side of the system of linear equations
    # A = the left hand matrix of linear equations
    # Solve the system of linear equations Ax = b
    stationary_distribution = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # linalg.lstsq returns multiple outputs; index 0 is x

    return stationary_distribution

# Irreducible and Aperiodic Markov Chains
mk1 = [
    [0.2, 0.2, 0.6],
    [0.1, 0.8, 0.1],
    [0.1, 0.2, 0.7]
]

mk2 = [
    [0.4, 0.4, 0.2],
    [0.3, 0.5, 0.2],
    [0.5, 0.4, 0.1]
]


# Simulate a Markov chain and estimate the stationary distribution
def simulate_markov_chain(transition_matrix, steps, initial_distribution=None):
    # Ensure the matrix is a numpy array
    P = np.array(transition_matrix)
    
    # Initializing vars
    num_states = P.shape[0]
    visits = np.zeros(num_states)   # vector to keep track of # of visits

    # Set initial distribution: uniform if not provided
    if initial_distribution is None:
        initial_distribution = np.ones(num_states) / num_states

    # Start state based on the initial distribution
    state = np.random.choice(num_states, p=initial_distribution)

    # sum of visits to each to each state after n steps
    for i in range(steps):
        visits[state] += 1
        state = np.random.choice(num_states, p=P[state])

    # estimated distribution is visits / steps
    estimated_distribution = visits / steps
    return estimated_distribution


# Exact stationary distribution
#exact_pi = compute_stationary_distribution(mk1)
#print("Exact stationary distribution:", exact_pi)

# Simulated stationary distribution
# id = np.array([1, 0, 0])
mkc = mk1       # markov chain to simulate

steps = 100  # Number of steps for simulation
simulated_pi = simulate_markov_chain(mkc, steps)
print(f"Simulated stationary distribution after {steps} steps:", simulated_pi)

steps = 500  # Number of steps for simulation
simulated_pi = simulate_markov_chain(mkc, steps)
print(f"Simulated stationary distribution after {steps} steps:", simulated_pi)

steps = 1000  # Number of steps for simulation
simulated_pi = simulate_markov_chain(mkc, steps)
print(f"Simulated stationary distribution after {steps} steps:", simulated_pi)

steps = 5000  # Number of steps for simulation
simulated_pi = simulate_markov_chain(mkc, steps)
print(f"Simulated stationary distribution after {steps} steps:", simulated_pi)

steps = 10000  # Number of steps for simulation
simulated_pi = simulate_markov_chain(mkc, steps)
print(f"Simulated stationary distribution after {steps} steps:", simulated_pi)

# Compute stationary distributions
stationary_1 = compute_stationary_distribution(mk1)
stationary_2 = compute_stationary_distribution(mk2)

print("Stationary distribution for Markov chain 1:", stationary_1)
print("Stationary distribution for Markov chain 2:", stationary_2)


