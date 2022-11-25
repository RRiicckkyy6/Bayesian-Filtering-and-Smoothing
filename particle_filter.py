# Bootstrap filter: 
# a special case of the general particle filter where the dynamic model p(x_k|x_k-1)
# is used as the importance distribution
# tested on a simple pendulum model

# test() computes the particle (bootstrap) filter estimates

import numpy as np
import matplotlib.pyplot as plt
import math

# draw N samples from the prior distribution p(x0), assuming that this is Gaussian
# N number of particles
def sample_prior(m0,P0,N):
    d = m0.size
    X0 = np.zeros((N,d,1))
    for i in range(N):
        x = np.random.multivariate_normal(np.reshape(m0,d),P0)
        X0[i] = np.reshape(x,(d,1))
    return X0

def non_linear_function_st(x):
    damp = 0.005
    f1 = x[0] + x[1] * 0.01
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

# draw a sample from the importance distribution (state transition distribution)
def sample_imp(x_k_1,Q,N):
    d = x_k_1[0].size
    x_k = np.zeros((N,d,1))
    for i in range(N):
        x_k[i] = non_linear_function_st(x_k_1[i])
        x_k[i] = x_k[i] + np.reshape(np.random.multivariate_normal(np.zeros(d),Q),(d,1))
    return x_k

# update new particle weights
def weight_up(w_k_1,y_k,x_k,R,N):
    w_k = np.zeros(N)
    for i in range(N):
        w_k[i] = w_k_1[i] * math.exp(-(non_linear_function_m(x_k[i])-y_k)**2/R/2)
    return w_k

# Perform resampling
def resample(par,w):
    N = w.size
    d = par[0].size
    re = np.random.choice(N,N,p=w)
    temp = np.zeros((N,d,1))
    for i in range(N):
        temp[i] = par[re[i]]
    return temp

def generate_test(T,N):
    # Discretized pendulum tracking
    d = 2
    Q = np.zeros((T,2,2))
    R = np.zeros((T,1,1))
    q = 0.1
    t = 0.01
    Q0 = np.array([[q*t*t*t/3,q*t*t/2],[q*t*t/2,q*t]])
    R0 = np.array([[1]])
    for i in range(T):
        Q[i] = Q0
        R[i] = R0
    m0 = np.zeros((d,1))
    P0 = np.zeros((d,d))
    P0[0][0] = 1
    P0[1][1] = 1
    X, Y = simulation(Q,R,m0,P0,T)

    # initialization
    XXX = np.zeros((T,N,d,1))
    WWW = np.zeros((T,N))
    X0 = sample_prior(m0,P0,N)
    W0 = np.ones(N) / N
    XXX[0] = X0
    WWW[0] = W0
    pre_M = np.zeros((T,d,1))
    pre_V = np.zeros((T,d,d))
    pre_M[0] = m0
    pre_V[0] = P0
    # sample importance distribution
    for i in range(T-1):
        XXX[i+1] = sample_imp(XXX[i],Q[i],N)
        WWW[i+1] = weight_up(WWW[i],Y[i+1],XXX[i+1],R[i+1],N)
        W_sum = 0
        for j in range(N):
            W_sum = W_sum + WWW[i+1][j]
        WWW[i+1] = WWW[i+1] / W_sum
        re = resample(XXX[i+1],WWW[i+1])
        XXX[i+1] = re
        WWW[i+1] = np.ones(N)/N
        # compute estimate of the expected value of the state
        temp = np.zeros((d,1))
        for m in range(N):
            temp = temp + re[m]
        temp = temp / N
        pre_M[i+1] = temp
        # compute estimate of the variance of the state.
        tem = np.zeros((d,d))
        for a in range(N):
            tem = tem + np.dot((re[a] - pre_M[i+1]),(re[a] - pre_M[i+1]).T)
        tem = tem / N
        pre_V[i+1] = tem
    return X, Y, pre_M, pre_V

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
    

def plot_data(x, y, pre, v, ee):
    N = x.shape[0]
    d1 = x[0].size
    d2 = y[0].size
    X = np.reshape(x,(N,d1))
    Y = np.reshape(y,(N,d2))
    filter_pre = np.reshape(pre,(N,d1))
    T = np.arange(N)
    V = v[:,0,0]
    V = np.reshape(V,N)
    V = np.sqrt(V)
    filter_pre_pl = filter_pre[:,0]
    filter_plot = np.reshape(filter_pre_pl,N)
    u_confidence = filter_pre_pl + 2 * V
    l_confidence = filter_pre_pl - 2 * V
    plt.plot(T,X[:,0],'--',label='states')
    plt.plot(T[1:],Y[1:],'.',label='observations')
    plt.plot(T,filter_plot, label='filter predictions, rmse = {e:.3f}'.format(e=ee))
    plt.plot(T,u_confidence,':r',label='95% confidence interval')
    plt.plot(T,l_confidence,':r')
    plt.title("Particle Filter (Bootstrap Filter)")
    plt.xlabel("time step")
    plt.legend()
    plt.show()

def test():
    T = 1000
    N = 600
    x,y,m,v = generate_test(T,N)
    error = rmse(m,x)
    plot_data(x,y,m,v,error)

def rmse(pre,state):
    N = pre.size
    r_m_s_e = np.linalg.norm(pre-state) / np.sqrt(N)
    return r_m_s_e

test()