import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls


# np.random.seed(0)

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

    print()
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
        # r = y - np.dot(As, z)
        
        r_p = y - np.dot(A, x_hat)
        print("r:",np.linalg.norm(r_p),"\ti:",i,"\tcount:",count)
        r = y - np.dot(A, x_hat)
        count+=1

        
    return x_hat, S




# Parameters
M = 60 #Number of measurements
curr_sample=0
N = 100  # Size of each measurement
p = 0.02  # Mean of the Bernoulli distribution


#Load the data (the N*1 sparse vector)
# ith value of 0: not infected and 1: infected
#group_data M samples of size 100 soldiers
group_data=loadmat('/home/frank/Documents/git/Boolean_compressed_Sensing/GroupTesting.mat')['x'][curr_sample,:N]

Y,A=construct_measurements_randompool(group_data, M, p,N)

print("Y:",Y)
z,S=nnomp(10**-40,Y,A)
# print("z:",z)
print(np.sort(list(S)))
print(np.nonzero(group_data))
