# Basic Kalman smoother: 
# Closed form solution to the Bayesian smoothing equation of linear Gaussian State Space Models
# RTS smoother implementation: 
# first run the Kalman filter, then use backward recursion to compute p(x_k|y_1_T) from p(x_k_1|y_1_T)

# test_1d() 1d Kalman Filter
# test_2d() multi-dimensional Kalman Filter
import numpy as np
import matplotlib.pyplot as plt
import math


# Terminology
# A_k transition matrix of dynamic model at step k
# H_k measurement model matrix at step k
# Q_k process noise at step k
# R_k measurement noise at step k
# P_k covariance matrix of filtering distribution at step k
# m_k mean vector of filtering distribution at step k

# Prediction step
def prediction(A_k_1, P_k_1, m_k_1, Q_k_1):
    pre_m = np.dot(A_k_1, m_k_1)
    pre_cov = np.dot(np.dot(A_k_1,P_k_1),A_k_1.transpose()) + Q_k_1
    return pre_m, pre_cov

# Update step
def update(y_k, H_k, pre_m, pre_P, R_k):
    v_k = y_k - np.dot(H_k, pre_m)
    S_k = np.dot(np.dot(H_k,pre_P),H_k.transpose()) + R_k
    K_k = np.dot(np.dot(pre_P,H_k.transpose()),np.linalg.inv(S_k))
    m_k = pre_m + np.dot(K_k, v_k)
    P_k = pre_P - np.dot(np.dot(K_k,S_k),K_k.transpose())
    return m_k, P_k


# Simulate the process
def simulation(A, H, Q, R, m0, P0, N):
    # if m0.shape == (1,) and R[0].size == 1:
    #     x0 = np.random.multivariate_normal(m0,P0)
    #     X = np.zeros((N,1,1))
    #     X[0] = np.array([[x0]])
    #     Y = np.zeros((N,1,1))
    #     Y[0] = np.array([[0]])
    #     for i in range(N-1):
    #         X[i+1] = np.dot(A[i],X[i]) + np.random.multivariate_normal(np.array([0]),Q[i])
    #         Y[i+1] = np.dot(H[i+1],X[i+1]) + np.random.multivariate_normal(np.array([0]),R[i+1])
    # elif m0.shape == (1,) and R[0].size != 1:
    #     x0 = np.random.normal(m0,P0)
    #     X = np.zeros((N,1,1))
    #     X[0] = np.array([[x0]])
    #     m = int(np.sqrt(R[0].size))
    #     Y = np.zeros((N,m,1))
    #     Y[0] = np.zeros((m,1))
    #     for i in range(N-1):
    #         X[i+1] = np.dot(A[i],X[i]) + np.random.multivariate_normal(np.array([0]),Q[i])
    #         Y[i+1] = np.dot(H[i+1],X[i+1]) + np.random.multivariate_normal(np.zeros(m),R[i+1])
    # elif m0.shape != (1,) and R[0].size == 1:
    #     n = m0.size
    #     x0 = np.random.multivariate_normal(np.reshape(m0,n),P0) 
    #     X = np.zeros((N,n,1))
    #     X[0] = np.reshape(x0,(n,1))
    #     Y = np.zeros((N,1,1))
    #     Y[0] = np.array([[0]])
    #     for i in range(N-1):
    #         X[i+1] = np.dot(A[i],X[i]) + np.random.multivariate_normal(np.zeros(n),Q[i])
    #         Y[i+1] = np.dot(H[i+1],X[i+1]) + np.random.multivariate_normal(np.array([0]),R[i+1])
    
    n = m0.size
    m = int(np.sqrt(R[0].size))
    x0 = np.random.multivariate_normal(np.reshape(m0,n),P0)
    X = np.zeros((N,n,1))
    X[0] = np.reshape(x0,(n,1))
    Y = np.zeros((N,m,1))
    Y[0] = np.zeros((m,1))
    for i in range(N-1):
        X[i+1] = np.dot(A[i],X[i]) + np.reshape(np.random.multivariate_normal(np.zeros(n),Q[i]),(n,1))
        Y[i+1] = np.dot(H[i+1],X[i+1]) + np.reshape(np.random.multivariate_normal(np.zeros(m),R[i]),(m,1))
    return X, Y

# root mean squared error
def rmse(pre,state):
    N = pre.size
    r_m_s_e = np.linalg.norm(pre-state) / np.sqrt(N)
    return r_m_s_e



def generate_1d_test(N):
    # Gaussian random walk
    A0 = np.array([[1]])
    H0 = np.array([[1]])
    A = np.zeros((N,1,1))
    H = np.zeros((N,1,1))
    Q = np.zeros((N,1,1))
    R = np.zeros((N,1,1))
    sigma_q = np.array([[1]]) # process noise parameters
    sigma_r = np.array([[1]]) # measurement noise parameters
    for i in range(N):
        A[i] = A0
        H[i] = H0
        Q[i] = sigma_q
        R[i] = sigma_r
    m0 = np.array([[0]])
    P0 = np.array([[1]])
    X, Y = simulation(A,H,Q,R,m0,P0,N)
    T = np.arange(N)
    # Kalman filter applied to the observations Y
    pre_mean = np.zeros((N,1,1))
    pre_cov = np.zeros((N,1,1))
    filter_mean = np.zeros((N,1,1))
    filter_cov = np.zeros((N,1,1))
    filter_mean[0] = m0
    filter_cov[0] = P0
    for i in range(N-1):
        pre_mean[i+1], pre_cov[i+1] = prediction(A[i],filter_cov[i],filter_mean[i],Q[i])
        filter_mean[i+1], filter_cov[i+1] = update(Y[i+1],H[i+1],pre_mean[i+1],pre_cov[i+1],R[i+1])
    return filter_mean, filter_cov, X, Y

def plot_data_1d(mean, cov, x, y):
    N = x.shape[0]
    T = np.arange(N)
    X = np.reshape(x,N)
    Y = np.reshape(y,N)
    filter_mean = np.reshape(mean,N)
    filter_cov = np.reshape(cov,N)
    u_confidence = filter_mean + 2 * filter_cov
    l_confidence = filter_mean - 2 * filter_cov
    plt.plot(T,X,'--',label='states')
    plt.plot(T[1:],Y[1:],'.',label='observations')
    plt.plot(T,filter_mean, label='filter predictions')
    plt.plot(T,u_confidence,':r',label='95% confidence interval')
    plt.plot(T,l_confidence,':r')
    plt.title("Kalman filter for 1D state and measurement space")
    plt.xlabel("time step k")
    plt.legend()
    plt.show()

def plot_data_2d(mean, cov, x, y):
    N = x.shape[0]
    X = np.reshape(x,(N,4))
    Y = np.reshape(y,(N,2))
    filter_mean = mean
    plt.plot(X[:,0],X[:,1],'--',label='states')
    plt.plot(Y[1:,0],Y[1:,1],'.',label='observations')
    plt.plot(filter_mean[:,0],filter_mean[:,1], label='filter predictions')
    plt.legend()
    plt.show()

def generate_2d_test(N):
    # Discretized car tracking
    A0 = np.array([[1,0,0.1,0],[0,1,0,0.1],[0,0,1,0],[0,0,0,1]])
    H0 = np.array([[1,0,0,0],[0,1,0,0]])
    A = np.zeros((N,4,4))
    H = np.zeros((N,2,4))
    Q = np.zeros((N,4,4))
    R = np.zeros((N,2,2))
    Q0 = np.array([[0.001/3,0,0.01/2,0],[0,0.001/3,0,0.01/2],[0.01/2,0,0.1,0],[0,0.01/2,0,0.1]])
    R0 = np.array([[0.25,0],[0,0.25]])
    for i in range(N):
        A[i] = A0
        H[i] = H0
        Q[i] = Q0
        R[i] = R0
    m0 = np.zeros((4,1))
    P0 = np.zeros((4,4))
    P0[0][0] = 0.1
    P0[1][1] = 0.1
    P0[2][2] = 0.1
    P0[3][3] = 0.1
    X, Y = simulation(A,H,Q,R,m0,P0,N)
    pre_mean = np.zeros((N,4,1))
    pre_cov = np.zeros((N,4,4))
    filter_mean = np.zeros((N,4,1))
    filter_cov = np.zeros((N,4,4))
    filter_mean[0] = m0
    filter_cov[0] = P0
    for i in range(N-1):
        pre_mean[i+1], pre_cov[i+1] = prediction(A[i],filter_cov[i],filter_mean[i],Q[i])
        filter_mean[i+1], filter_cov[i+1] = update(Y[i+1],H[i+1],pre_mean[i+1],pre_cov[i+1],R[i+1])
    return filter_mean, filter_cov, X, Y


def test_2d():
    N = 100
    m,c,x,y = generate_2d_test(N)
    plot_data_2d(m,c,x,y)
    print(rmse(m,x))

def test_1d():
    N = 100
    m,c,x,y = generate_1d_test(N)
    plot_data_1d(m,c,x,y)
    error = 0
    for i in range(20):
        m,c,x,y = generate_1d_test(N)
        error = error + rmse(m,x)
    print("averge rmse is: ",error/20)