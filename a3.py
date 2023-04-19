import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# define Beale function
def Beale(x):
    term1 = (1.5 - x[0] + x[0]*x[1])**2
    term2 = (2.25 - x[0] + x[0]*(x[1]**2))**2
    term3 = (2.625 - x[0] + x[0]*(x[1]**3))**2
    return term1+term2+term3

# using gradient descent to optimize Beale function
def Gradient_descent(x, lr, max_iter):
    num_iter = 0
    hist_fval = np.zeros((max_iter+1, 1))
    hist_fval[0] = Beale(x)
    while num_iter < max_iter:
        grad = np.zeros(len(x))
        for i in range(len(x)):
            delta = 0.01
            x_new = np.copy(x)
            x_new[i] += delta
            grad[i] = (Beale(x_new) - Beale(x)) / delta
        x -= lr * grad
        num_iter += 1
        hist_fval[num_iter] = Beale(x)
    return x, hist_fval

# using SGD to optimize Beale function
def SGD(x, batch_size, lr, max_iter):
    num_iter = 0
    hist_fval = np.zeros((max_iter+1, 1))
    hist_fval[0] = Beale(x)
    while num_iter < max_iter:
        rand_indices = np.random.choice(len(x), batch_size)
        rand_grad = np.zeros(len(x))
        for i in rand_indices:
            delta = 0.01
            x_new = np.copy(x)
            x_new[i] += delta
            rand_grad += (Beale(x_new) - Beale(x)) / delta
        grad = rand_grad / batch_size
        x -= lr * grad
        num_iter += 1
        hist_fval[num_iter] = Beale(x)
    return x, hist_fval

# using Adam to optimize Beale function
def Adam(x, lr, beta1, beta2, eps, max_iter):
    num_iter = 0
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    hist_fval = np.zeros((max_iter+1, 1))
    hist_fval[0] = Beale(x)
    while num_iter < max_iter:
        grad = np.zeros(len(x))
        for i in range(len(x)):
            delta = 0.01
            x_new = np.copy(x)
            x_new[i] += delta
            grad[i] = (Beale(x_new) - Beale(x)) / delta
        t = num_iter + 1
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad ** 2
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        x -= lr / (np.sqrt(v_hat) + eps) * m_hat
        num_iter += 1
        hist_fval[num_iter] = Beale(x)
    return x, hist_fval

# using Adagrad to optimize Beale function
def Adagrad(x, lr, eps, max_iter):
    num_iter = 0
    G = np.zeros(len(x))
    hist_fval = np.zeros((max_iter+1, 1))
    hist_fval[0] = Beale(x)
    while num_iter < max_iter:
        grad = np.zeros(len(x))
        for i in range(len(x)):
            delta = 0.01
            x_new = np.copy(x)
            x_new[i] += delta
            grad[i] = (Beale(x_new) - Beale(x)) / delta
            G[i] += grad[i] ** 2
        x -= lr / (np.sqrt(G) + eps) * grad
        num_iter += 1
        hist_fval[num_iter] = Beale(x)
    return x, hist_fval

# using RMSprop to optimize Beale function
def RMSprop(x, lr, gamma, eps, max_iter):
    num_iter = 0
    G = np.zeros(len(x))
    hist_fval = np.zeros((max_iter+1, 1))
    hist_fval[0] = Beale(x)
    while num_iter < max_iter:
        grad = np.zeros(len(x))
        for i in range(len(x)):
            delta = 0.01
            x_new = np.copy(x)
            x_new[i] += delta
            grad[i] = (Beale(x_new) - Beale(x)) / delta
            G[i] = gamma * G[i] + (1 - gamma) * grad[i] ** 2
        x -= lr / (np.sqrt(G) + eps) * grad
        num_iter += 1
        hist_fval[num_iter] = Beale(x)
    return x, hist_fval

# test the performance of Gradient descent, SGD, Adam, Adagrad and RMSprop
x_0 = np.array([1.0, 1.0]) # initialize x
max_iter = 100 # max iteration number
eps = 1e-8 # numerical stability parameter
lr = 0.01 # learning rate
gamma = 0.9 # decay rate for RMSprop
batch_size = 10 # batch size for SGD
beta1, beta2 = 0.9, 0.999 # decay rates for Adam
result_gd, hist_gd = Gradient_descent(x_0, lr, max_iter)
result_sgd, hist_sgd = SGD(x_0, batch_size, lr, max_iter)
result_adam, hist_adam = Adam(x_0, lr, beta1, beta2, eps, max_iter)
result_adagrad, hist_adagrad = Adagrad(x_0, lr, eps, max_iter)
result_rmsprop, hist_rmsprop = RMSprop(x_0, lr, gamma, eps, max_iter)
# plot the Beale function value by iteration number for different optimization algorithms
plt.plot(hist_gd, label='Gradient descent')
plt.plot(hist_sgd, label='SGD')
plt.plot(hist_adam, label='Adam')
plt.plot(hist_adagrad, label='Adagrad')
plt.plot(hist_rmsprop, label='RMSprop')
plt.legend(loc='best')
plt.ylabel('Beale function')
plt.xlabel('iteration')
plt.title('Optimizing Beale function with different algorithms')
plt.show()