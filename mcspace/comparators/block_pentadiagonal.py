import numpy as np

def solve_pent_diagonal_thomas(off_off_diag, off_diag, diag, grad):
    #Thomas Algorithm for BLock Pentadiagonal (see Benkert & Fischer, 2007)
    #grad is the negative of the actual rhs of the equation

    ntaxa = len(diag[0])
    num_bins = len(diag)
    zero_bl = np.zeros((ntaxa, ntaxa))
    K_matrix = np.zeros((num_bins, ntaxa, ntaxa))
    G_matrix = np.zeros((num_bins, ntaxa, ntaxa))
    Y_matrix = np.zeros((num_bins, ntaxa, ntaxa))
    Z_matrix = np.zeros((num_bins, ntaxa, ntaxa))
    A = np.concatenate(([zero_bl], off_off_diag))
    A = np.concatenate(([zero_bl], A))
    B = np.concatenate(([zero_bl], off_diag))
    C = diag
    D = np.concatenate((off_diag, [zero_bl]))
    E = np.concatenate((off_off_diag, [zero_bl]))
    E = np.concatenate((E, [zero_bl]))

    r = np.zeros((num_bins, ntaxa))
    for i in range(num_bins):
        if i==0:
            K_matrix[0]=B[0]
            G_matrix[0]=C[0]
            G_inv = np.linalg.inv(G_matrix[0])
            Y_matrix[0]=np.matmul(G_inv, D[0]) 
            Z_matrix[0]=np.matmul(G_inv, E[0])
            r[0]=np.matmul(G_inv, -grad[0])
        elif i==1:
            K_matrix[i]=B[1]
            G_matrix[i]=C[1]-np.matmul(K_matrix[i],Y_matrix[i-1])
            G_inv = np.linalg.inv(G_matrix[i])
            Y_matrix[i]=np.matmul(G_inv, D[i]-np.matmul(K_matrix[i],Z_matrix[i-1]))
            Z_matrix[i]=np.matmul(G_inv, E[i])
            r[i]=np.matmul(G_inv, -grad[i]-np.matmul(K_matrix[i],r[i-1]))
        else:
            K_matrix[i]=B[i]-np.matmul(A[i], Y_matrix[i-2])
            G_matrix[i]=C[i]-np.matmul(K_matrix[i],Y_matrix[i-1])-np.matmul(A[i],Z_matrix[i-2])
            G_inv = np.linalg.inv(G_matrix[i])
            Y_matrix[i]=np.matmul(G_inv, D[i]-np.matmul(K_matrix[i],Z_matrix[i-1]))
            Z_matrix[i]=np.matmul(G_inv, E[i])
            r[i]=np.matmul(G_inv, -grad[i]-np.matmul(A[i],r[i-2])-np.matmul(K_matrix[i],r[i-1]))
    mu = np.zeros((num_bins, ntaxa))
    mu[-1] = r[-1]
    mu[-2] = r[-2]-np.matmul(Y_matrix[-2], mu[-1])
    for i in reversed(range(num_bins-2)):
        mu[i] = r[i]-np.matmul(Y_matrix[i], mu[i+1])-np.matmul(Z_matrix[i], mu[i+2])
    return mu
def bound_eigenval(A):
    """
    Deals with numerical stability by lower bounding all eigenvalues of A at a normal threshold (1e-3)
    Parameters
    ----------
    A : some symmetric matrix
    Returns
    -------
    recalculated A

    """
    w, v = np.linalg.eigh(A)
    #print(A)
    #print(w)
    eig_mat = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        if w[i] < 1e-4:
            w[i]=1e-4
        if isinstance(w[i], complex):
            w[i]=1e-4
        eig_mat[i,i]=w[i]
    #eig mat is now the new diagonal matrix
    return v.dot(eig_mat).dot(v.T)

def solve_pent_diagonal_naive(off_off_diag, off_diag, diag, rhs):
    """
    Construct a large matrix, and then just invert it to get the solution
    """
    ntaxa = len(diag[0])
    num_bins = len(diag)
    zero_bl = np.zeros((ntaxa, ntaxa))
    rows = [[]]
    for i in range(num_bins):
        row = []  
        added = 0
        if i==0:
            added=3
            row.append(diag[i])
            row.append(off_diag[i])
            row.append(off_off_diag[i])
        elif i==1:
            added=4
            row.append(off_diag[i-1])
            row.append(diag[i])
            row.append(off_diag[i])
            row.append(off_off_diag[i])
        elif i==num_bins-2:
            added=4
            row.append(off_off_diag[i-2])
            row.append(off_diag[i-1])
            row.append(diag[i])
            row.append(off_diag[i])
        elif i==num_bins-1:
            added=3
            row.append(off_off_diag[i-2])
            row.append(off_diag[i-1])
            row.append(diag[i])
        else:
            added=5
            row.append(off_off_diag[i-2])
            row.append(off_diag[i-1])
            row.append(diag[i])
            row.append(off_diag[i])
            row.append(off_off_diag[i])
        for j in range(num_bins-added):
            row.append(zero_bl)
        rows.append(row)
    rows.pop(0)
    large_matrix = np.block(rows)
    large_matrix = bound_eigenval(large_matrix)
    answer =  np.matmul(np.linalg.inv(large_matrix), rhs)
    return answer.reshape((num_bins, ntaxa))