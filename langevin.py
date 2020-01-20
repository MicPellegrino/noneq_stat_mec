import math
import numpy as np
import matplotlib.pyplot as plt

# Function outputting a Wiener process from a standard gaussian vector
# N     number of steps for each process
# M     number of processes (only 1 by default)
# w_0   initial position (by default = 0)
def standard_wiener_process (N, w_0 = 0) :
    sgv = np.random.normal(0.0, 1.0, N)
    sgv[0] = w_0
    return np.cumsum(sgv)

# [a] generate a Wiener process

t_fin = 10.0
dt = 0.001
N_sam = math.ceil(t_fin/dt)
wp = np.sqrt(dt) * standard_wiener_process(N_sam)
t_vec = np.arange(0, t_fin, dt)

plt.figure(1)
plt.plot(t_vec, wp)
plt.xlim([0,t_fin])
plt.xlabel('t')
plt.ylabel('w(t)')
plt.title('Wiener process realization')
plt.show()

# [b] estimate velocity variance

gamma = np.array([0.1, 1.0])
Lambda = np.array([1.0, 2.0])
v_0 = 10.0   # Otherwise it has no sense at all!

alpha_0 = -dt*gamma+1
alpha_1 = np.sqrt( dt*Lambda )
print(alpha_0)
print(alpha_1)

M_avg = 1000

v_0 = 1.0
v1_matrix = np.zeros((N_sam, M_avg), dtype=np.double)
v2_matrix = np.zeros((N_sam, M_avg), dtype=np.double)

v1_matrix[ 0, : ] = v_0
v2_matrix[ 0, : ] = v_0
for k in range(0, N_sam-1) :
    v1_matrix[ k+1, : ] = alpha_0[0]*v1_matrix[ k, : ] + alpha_1[0]*np.random.normal(0.0, 1.0, M_avg)
    v2_matrix[ k+1, : ] = alpha_0[1]*v2_matrix[ k, : ] + alpha_1[1]*np.random.normal(0.0, 1.0, M_avg)

v1_mean = v1_matrix.mean(axis=1)
v2_mean = v2_matrix.mean(axis=1)

plt.figure(2)
plt.plot(t_vec, v1_mean, 'b-', label='gamma = 0.3, lambda = 1.0')
plt.plot(t_vec, v2_mean, 'r-', label='gamma = 1.0, lambda = 2.0')
plt.xlim([0,t_fin])
plt.xlabel('t')
plt.ylabel('<v(t)>')
plt.title('Wiener process realization')
plt.legend()
plt.show()

