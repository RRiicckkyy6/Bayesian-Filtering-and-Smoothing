# Extended Kalman filter: 
# linear approximation to the Bayesian filtering equation of nonlinear state space models
# linearization perfomred with first order Taylor series expansion
# code tested with a simple discretization of the pendulum model

# test() runs both Extended Kalman Filter and Statistically Linearized Kalman Filter on the same synthetic dataset
import numpy as np
import matplotlib.pyplot as plt
import math

# Terminology
# f_k evaluation of state transition function of dynamic model using m_k
# H_k evaluation of measurement function using m_k
# Q_k process noise at step k
# R_k measurement noise at step k
# P_k covariance matrix of filtering distribution at step k
# m_k mean vector of filtering distribution at step k

# Prediction step
def prediction(m_k_1, P_k_1, Q_k_1):
    pre_m = non_linear_function_st(m_k_1)
    pre_cov = np.dot(np.dot(hessian_st(m_k_1),P_k_1),hessian_st(m_k_1).transpose()) + Q_k_1
    return pre_m, pre_cov

# Update step
def update(y_k, pre_m, pre_P, R_k):
    # Need an extra if statement for the case of scalar state space but vector measurement space
    v_k = y_k - non_linear_function_m(pre_m)
    S_k = np.dot(np.dot(hessian_m(pre_m),pre_P),hessian_m(pre_m).transpose()) + R_k
    K_k = np.dot(np.dot(pre_P,hessian_m(pre_m).transpose()),np.linalg.inv(S_k))
    m_k = pre_m + np.dot(K_k, v_k)
    # P_k = pre_P - np.outer(K_k*S_k,K_k.transpose())
    P_k = pre_P - np.dot(np.dot(K_k,S_k),K_k.transpose())
    return m_k, P_k

def prediction_statistical(m_k_1, P_k_1, Q_k_1):
    pre_m = expectation1(m_k_1,P_k_1)
    pre_cov = np.dot(np.dot(cross_expectation1(m_k_1,P_k_1),np.linalg.inv(P_k_1)),cross_expectation1(m_k_1,P_k_1).transpose()) + Q_k_1
    return pre_m, pre_cov


def update_statistical(y_k, pre_m, pre_P, R_k):
    # Need an extra if statement for the case of scalar state space but vector measurement space
    v_k = y_k - expectation2(pre_m, pre_P)
    S_k = np.dot(np.dot(cross_expectation2(pre_m, pre_P),np.linalg.inv(pre_P)),cross_expectation2(pre_m, pre_P).transpose()) + R_k
    K_k = np.dot(cross_expectation2(pre_m, pre_P).transpose(),np.linalg.inv(S_k))
    m_k = pre_m + np.dot(K_k, v_k)
    P_k = pre_P - np.dot(np.dot(K_k,S_k),K_k.transpose())
    return m_k, P_k

def expectation1(m_k_1, P_k_1):
    E = np.zeros((2,1))
    damp = 0.005
    E[0] = m_k_1[0] + m_k_1[1] * 0.01
    E[1] = (1-damp) * m_k_1[1] - 9.81 * math.sin(m_k_1[0]) * math.exp(-P_k_1[0][0]/2) * 0.01
    return E

def expectation2(pre_m, pre_P):
    E = math.sin(pre_m[0]) * math.exp(-pre_P[0][0]/2)
    return np.reshape(E,(1,1))

def cross_expectation1(m_k_1, P_k_1):
    E = np.zeros((2,2))
    damp = 0.005
    E[0][0] = P_k_1[0][0] + 0.01 * P_k_1[0][1]
    E[0][1] = P_k_1[0][1] + 0.01 * P_k_1[1][1]
    E[1][0] = (1 - damp) * P_k_1[0][1] - 9.81 * 0.01 * math.cos(m_k_1[0]) * P_k_1[0][0] * math.exp(-P_k_1[0][0]/2)
    E[1][1] = (1 - damp) * P_k_1[1][1] - 9.81 * 0.01 * math.cos(m_k_1[0]) * P_k_1[0][1] * math.exp(-P_k_1[0][0]/2)
    return E

def cross_expectation2(pre_m, pre_P):
    E = np.zeros((2,1))
    E[0][0] = math.cos(pre_m[0]) * pre_P[0][0] * math.exp(-pre_P[0][0]/2)
    E[1][0] = math.cos(pre_m[0]) * pre_P[0][1] * math.exp(-pre_P[0][0]/2)
    return E.T

def non_linear_function_st(x):
    f1 = x[0] + x[1] * 0.01
    damp = 0.005
    f2 = (1 - damp) * x[1] - 9.81 * math.sin(x[0]) * 0.01
    f = np.zeros(2)
    f[0] = f1
    f[1] = f2
    f = np.array(f)
    f = np.reshape(f,(2,1))
    return f 

def non_linear_function_m(x):
    h = math.sin(x[0])
    h = np.array([[h]])
    return h

def hessian_st(x):
    F = np.zeros((2,2))
    damp = 0.005
    F[0][0] = 1
    F[0][1] = 0.01
    F[1][0] = -9.81 * math.cos(x[0]) * 0.01
    F[1][1] = 1 - damp
    return F

def hessian_m(x):
    F = np.zeros((1,2))
    F[0][0] = math.cos(x[0])
    F[0][1] = 0
    return F

# Simulate the process
def simulation(Q, R, m0, P0, N):
    # if (m0.shape == (1,)) and (R[0].size == 1):
    #     x0 = np.random.normal(m0,P0)
    #     X = np.zeros((N,1,1))
    #     X[0] = np.array([[x0]])
    #     Y = np.zeros((N,1,1))
    #     Y[0] = np.array([[0]])
    #     for i in range(N-1):
    #         X[i+1] = non_linear_function_st(X[i]) + np.random.normal(0,Q[i])
    #         Y[i+1] = non_linear_function_m(X[i+1]) + np.random.normal(0,R[i+1])
    # elif (m0.shape == (1,)) and (R[0].size != 1):
    #     x0 = np.random.normal(m0,P0)
    #     X = np.zeros((N,1,1))
    #     X[0] = np.array([[x0]])
    #     m = int(np.sqrt(R[0].size))
    #     Y = np.zeros((N,m,1))
    #     Y[0] = np.zeros((m,1))
    #     for i in range(N-1):
    #         X[i+1] = non_linear_function_st(X[i]) + np.random.normal(0,Q[i])
    #         Y[i+1] = non_linear_function_m(X[i+1]) + np.reshape(np.random.multivariate_normal(np.zeros(m),R[i]),(m,1))
    # elif (m0.shape != (1,)) and (R[0].size == 1):
    #     n = m0.size
    #     x0 = np.random.multivariate_normal(np.reshape(m0,n),P0)
    #     X = np.zeros((N,n,1))
    #     X[0] = np.reshape(x0,(n,1))
    #     Y = np.zeros((N,1,1))
    #     Y[0] = np.array([[0]])
    #     for i in range(N-1):
    #         X[i+1] = non_linear_function_st(X[i]) + np.reshape(np.random.multivariate_normal(np.zeros(n),Q[i]),(n,1))
    #         Y[i+1] = non_linear_function_m(X[i+1]) + np.random.normal(0,R[i+1])
    n = m0.size
    m = int(np.sqrt(R[0].size))
    x0 = np.random.multivariate_normal(np.reshape(m0,n),P0)
    X = np.zeros((N,n,1))
    X[0] = np.reshape(x0,(n,1))
    Y = np.zeros((N,m,1))
    Y[0] = np.zeros((m,1))
    for i in range(N-1):
        X[i+1] = non_linear_function_st(X[i]) + np.reshape(np.random.multivariate_normal(np.zeros(n),Q[i]),(n,1))
        Y[i+1] = non_linear_function_m(X[i+1]) + np.reshape(np.random.multivariate_normal(np.zeros(m),R[i]),(m,1))
    return X, Y



# root mean squared error
def rmse(pre,state):
    N = pre.size
    r_m_s_e = np.linalg.norm(pre-state) / np.sqrt(N)
    return r_m_s_e

def plot_data(mean, cov, x, y, mean2, cov2, e1, e2):
    N = x.shape[0]
    X = np.reshape(x,(N,2))
    Y = np.reshape(y,(N,1))
    filter_mean = np.reshape(mean,(N,2))
    filter_cov = cov[:,0,0]
    filter_cov = np.reshape(filter_cov,(N,1))
    print(filter_cov[0:50])
    X0 = X[:,0]
    f_mean0 = filter_mean[:,0]
    f_mean0 = np.reshape(f_mean0,(N,1))
    u_confidence = f_mean0 + 2 * np.sqrt(filter_cov)
    l_confidence = f_mean0 - 2 * np.sqrt(filter_cov)
    T = np.arange(N)
    plt.plot(T,X0,'--',label='states')
    plt.plot(T[1:],Y[1:],'.',label='observations')
    plt.plot(T,f_mean0, label='filter predictions, rmse = {e:.3f}'.format(e = e1))
    plt.plot(T,u_confidence,':r',label='95% confidence interval')
    plt.plot(T,l_confidence,':r')
    plt.xlabel("time step")
    plt.title("Extended Kalman Filter")
    plt.legend()
    plt.show()

    N = x.shape[0]
    X = np.reshape(x,(N,2))
    Y = np.reshape(y,(N,1))
    filter_mean = np.reshape(mean2,(N,2))
    filter_cov = cov2[:,0,0]
    filter_cov = np.reshape(filter_cov,(N,1))
    print(filter_cov[0:50])
    X0 = X[:,0]
    f_mean0 = filter_mean[:,0]
    f_mean0 = np.reshape(f_mean0,(N,1))
    u_confidence = f_mean0 + 2 * np.sqrt(filter_cov)
    l_confidence = f_mean0 - 2 * np.sqrt(filter_cov)
    T = np.arange(N)
    plt.plot(T,X0,'--',label='states')
    plt.plot(T[1:],Y[1:],'.',label='observations')
    plt.plot(T,f_mean0, label='filter predictions, rmse = {e:.3f}'.format(e = e2))
    plt.plot(T,u_confidence,':r',label='95% confidence interval')
    plt.plot(T,l_confidence,':r')
    plt.xlabel("time step")
    plt.title("Statistiaclly Learized Kalman Filter")
    plt.legend()
    plt.show()

def generate_test(N):
    # Discretized pendulum tracking
    Q = np.zeros((N,2,2))
    R = np.zeros((N,1,1))
    q = 0.1
    t = 0.01
    Q0 = np.array([[q*t*t*t/3,q*t*t/2],[q*t*t/2,q*t]])
    R0 = np.array([[1]])
    for i in range(N):
        Q[i] = Q0
        R[i] = R0
    m0 = np.zeros((2,1))
    P0 = np.zeros((2,2))
    P0[0][0] = 1
    P0[1][1] = 1
    X, Y = simulation(Q,R,m0,P0,N)
    pre_mean1 = np.zeros((N,2,1))
    pre_cov1 = np.zeros((N,2,2))
    filter_mean1 = np.zeros((N,2,1))
    filter_cov1 = np.zeros((N,2,2))
    filter_mean1[0] = m0
    filter_cov1[0] = P0

    pre_mean2 = np.zeros((N,2,1))
    pre_cov2 = np.zeros((N,2,2))
    filter_mean2 = np.zeros((N,2,1))
    filter_cov2 = np.zeros((N,2,2))
    filter_mean2[0] = m0
    filter_cov2[0] = P0


    # Extended Filter
    for i in range(N-1):
        pre_mean1[i+1], pre_cov1[i+1] = prediction(filter_mean1[i], filter_cov1[i] ,Q[i])
        filter_mean1[i+1], filter_cov1[i+1] = update(Y[i+1],pre_mean1[i+1],pre_cov1[i+1],R[i+1])
    
    # Statistical Linearized Filter
    for i in range(N-1):
        pre_mean2[i+1], pre_cov2[i+1] = prediction_statistical(filter_mean2[i], filter_cov2[i] ,Q[i])
        filter_mean2[i+1], filter_cov2[i+1] = update_statistical(Y[i+1],pre_mean2[i+1],pre_cov2[i+1],R[i+1])

    return filter_mean1, filter_cov1, X, Y, filter_mean2, filter_cov2



def test():
    N = 1000
    error1 = 0
    error2 = 0
    for i in range(20):
        m,c,x,y,m2,c2 = generate_test(N)
        error1 = error1 + rmse(m,x)
        error2 = error2 + rmse(m2,x)
    plot_data(m,c,x,y,m2,c2,error1/20,error2/20)
    print(rmse(m,x))