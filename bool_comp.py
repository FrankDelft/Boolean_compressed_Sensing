import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import cvxpy as cp

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

def basis_pursuit(y, A):
    # n is the number of variables in the signal x we want to recover
    n = A.shape[1]
    
    # Define the optimization variable
    x = cp.Variable(n)
    
    # Define the objective function (minimize the l1 norm of x)
    objective = cp.Minimize(cp.norm(x, 1))
    
    # Define the constraints (Ax = y)
    constraints = [A @ x == y,x>=0,x<=1]
    
    # Define the problem
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    problem.solve(solver=cp.CVXOPT)
    
    if x.value is not None:  # Check if the solution exists
        x_hat = x.value
        # Apply threshold if necessary
        threshold = 10**-5
        x_hat = np.where(x_hat < threshold, 0, x_hat)  # This replaces values below threshold with 0
        x_hat = np.where(x_hat >= threshold, 1, x_hat)  # This sets values above or equal to threshold to 1
    else:
        print("Solution not found or problem is infeasible.")
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
M = 1000 #Number of measurements
N = 100  # Size of each measurement
p = 0.0007  # Mean of the Bernoulli distribution
#Load the data (the N*1 sparse vector)
# ith value of 0: not infected and 1: infected
#group_data M samples of size 100 soldiers
group_data=loadmat('/home/frank/Documents/git/Boolean_compressed_Sensing/GroupTesting.mat')['x'][:,:N]
total_hamming_omp=0
total_hamming_basis=0




for i,s in enumerate(group_data):

    Y,A=construct_measurements_randompool(s, M, p,N)

    z_basis=basis_pursuit(Y,A)
    z_omp,S=nnomp(10**-40,Y,A)
    z_omp=np.ceil(z_omp)
    # print("Group_data:",np.nonzero(s))
    # print("z:",np.sort(list(S)))
    # print("Hamming distance basis:", hamming_distance(s, z_basis), "\ttotal non zero:",len(np.nonzero(s)[0]))
    # print("Hamming distance:", hamming_distance(s, z), "\ttotal non zero:",len(np.nonzero(s)[0]))
    total_hamming_omp+=hamming_distance(s, z_omp)
    total_hamming_basis+=hamming_distance(s, z_basis)
    
print("Total hamming distance:",total_hamming_basis)
print("Total hamming distance:",total_hamming_omp)