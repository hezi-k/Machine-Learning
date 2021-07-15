import numpy as np

matrixA = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
matrixB = np.array([[7, 8, 9], [10, 11, 12]], np.int32)
matrixC = np.array([[13, 14], [15, 16], [17, 18]], np.int32)
matrixD = np.array([[13, 14], [15, 16]], np.int32)

#This function adds two matrices together
def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return problem1(np.dot(A,B), C)

def problem3 (A, B, C):
    return A*B + C.transpose() 

def problem4 (x, S, y):
    return np.dot(np.dot(x.transpose(),y), S)

def problem5 (A):
    return np.zeros(A.shape)

def problem6 (A):
    return np.ones(A.shape)

def problem7 (A, alpha):

    num_rows = np.shape(A)[0]
    identityMatrix = np.eye(num_rows)

    return A + (np.dot(alpha, identityMatrix))

def problem8 (A, i, j):
    return A[i][j]

print(problem8(matrixB, 1, 1))

def problem9 (A, i):
    return np.sum(A[i])

def problem10 (A, c, d):
    if d > c:
        return np.mean(A[np.nonzero(np.logical_and(A >= c, A <= d))])
    elif c > d:
        return np.mean(A[np.nonzero(np.logical_and(A <= c, A >= d))])

def problem11 (A, k):
    eigValues = np.linalg.eig(A)[1]
    resultCol = A - k
    return eigValues[:, resultCol:]

def problem12 (A, x):
    return np.linalg.solve(A,x)

def problem13 (A, x):
    firstOperation = np.linalg.solve(A.transpose(),x.transpose())
    return firstOperation.tranpsosed()

