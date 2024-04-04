import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import cvxpy as cp
import seaborn as sns

np.random.seed(0)

#Function the generate the measurements based on random pooling using the binomial distribution
def construct_measurements_randompool(x, M, p,N):
    """
    Constructs measurements based on the given parameters.

    Parameters:
    x (array-like): The input data.
    M (int): The number of measurements to construct.
    p (float): The binomial probability.

    Returns:
    array-like: The constructed measurements.

    """
    # Now let's define the A matrix as a 1000*100 matrix
    A = np.random.binomial(1, p, size=(M, N))
    # Construct the measurements
    X = np.tile(x, (M, 1))
    Y = np.logical_or.reduce(A * X, axis=1).astype(int)
    return Y,A

def nnomp(tolerance, y, A):
    r = y.copy()
    S = set()
    n = A.shape[1]
    x_hat = np.zeros(n)

    count=0
    while np.linalg.norm(r) > tolerance and len(S) < A.shape[0]:
        # Compute correlation with residual
        h = np.dot(A.T, r)
                
        # Find index of maximum correlation
        i = np.argmax(h)
        
        # Avoid adding the same index to S
        if i in S:
            break
        
        S.add(i)  # Add this index to the support set
        
        # Solve non-negative least squares problem for current support set
        As = A[:, list(S)]
        z, _ = nnls(As, y)
        
        # Update x_hat with the current non-negative solution
        x_hat[list(S)] = z
        
        # Update residual
        r = y - np.dot(A, x_hat)
        count+=1

        
    return x_hat, S

def basis_pursuit(y, A,c):
    # n is the number of variables in the signal x we want to recover
    n = A.shape[1]
    
    # Define the optimization variable
    x = cp.Variable(n)
    
    # Define the objective function (minimize the l1 norm of x)
    objective = cp.Minimize(cp.norm(x, 1))
    
    # Define the constraints (Ax = y)
    constraints = [cp.norm(A @ x - y, 2) <= c, x >= 0, x <= 1]
    
    # Define the problem
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    problem.solve(solver=cp.SCS,verbose=False)
    
    if x.value is not None:  # Check if the solution exists
        x_hat = x.value
        # Apply threshold if necessary
        threshold = 10**-5
        x_hat = np.where(x_hat < threshold, 0, x_hat)  # This replaces values below threshold with 0
        x_hat = np.where(x_hat >= threshold, 1, x_hat)  # This sets values above or equal to threshold to 1
    else:
        print("Solution not found or problem is infeasible.")
        # print(y)
        x_hat = np.zeros(n)
    return x_hat

def hamming_distance(array1, array2):
    """
    Calculates the Hamming distance between two numpy arrays.

    Parameters:
    array1 (numpy.ndarray): The first array.
    array2 (numpy.ndarray): The second array.

    Returns:
    int: The Hamming distance between the two arrays.

    """
    if array1.shape != array2.shape:
        raise ValueError("Arrays must have the same shape.")
    
    distance = np.count_nonzero(array1 != array2)
    return distance




# Parameters

N = 100  # Size of each measurement
#Load the data (the N*1 sparse vector)
# ith value of 0: not infected and 1: infected
#group_data M samples of size 100 patients
group_data=loadmat('/home/frank/Documents/git/Boolean_compressed_Sensing/GroupTesting.mat')['x'][1:200,:N]

# Define the range of values for p and M
p_values =np.array([0.001,0.0025,0.005,0.01,0.05,0.1,0.2,0.4])
M_values = np.arange(10, 211, 20)

# Initialize an empty dictionary to store the hamming distances
hamming_distances_basis = { (p, M): 0 for p in p_values for M in M_values}
hamming_distances_omp = { (p, M): 0 for p in p_values for M in M_values}



# Create a list of p and M values
p_values = sorted(list(set([key[0] for key in hamming_distances_omp.keys()])))
M_values = sorted(list(set([key[1] for key in hamming_distances_omp.keys()])))


# Iterate over p and M values
for p in p_values:
    for M in M_values:
        print("p: ",p,"\t M:", M)
        # Initialize the total hamming distance for basis pursuit
        total_hamming_omp = 0
        total_hamming_basis = 0
        
        # Iterate over group_data
        for i, s in enumerate(group_data):
            Y, A = construct_measurements_randompool(s, M, p, N)
            z_omp = nnomp(10**-40,Y,A)[0]
            try:
                z_basis = basis_pursuit(Y, A, 1)
            except:
                print("failed")
                z_basis=np.zeros(N)
            z_omp = np.ceil(z_omp)
            total_hamming_omp += hamming_distance(s, z_omp)
            total_hamming_basis += hamming_distance(s, z_basis)
        
        # Store the total hamming distance for this p and M combination
        hamming_distances_omp[(p, M)] = total_hamming_omp
        hamming_distances_basis[(p, M)] = total_hamming_basis


# Create a matrix of hamming distances
hamming_matrix_omp = np.zeros((len(p_values), len(M_values)))
hamming_matrix_basis = np.zeros((len(p_values), len(M_values)))
for i, p in enumerate(p_values):
    for j, M in enumerate(M_values):
        hamming_matrix_omp[i, j] = hamming_distances_omp[(p, M)]
        hamming_matrix_basis[i, j] = hamming_distances_basis[(p, M)]




#Plot the heat map
sns.heatmap(hamming_matrix_omp, xticklabels=M_values, yticklabels=p_values)
plt.xlabel('M (Number of Measurements)')
plt.ylabel('p (Probability)')
plt.title('Total Hamming Distance for NNOMP')
plt.show()


# Plot the heat map
print(hamming_distances_basis)
sns.heatmap(hamming_matrix_basis, xticklabels=M_values, yticklabels=p_values)
plt.xlabel('M (Number of Measurements)')
plt.ylabel('p (Probability)')
plt.title('Total Hamming Distance for basis pursuit')
plt.show()
