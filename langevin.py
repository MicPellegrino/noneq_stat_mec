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

def variance_position_langevin (g, L, t) :
    a = np.exp(-g*t) - 1.0
    b = np.power( -np.exp(-g*t) + 1.0, 2 )
    return (L/(g**2)) * ( t + a/g - 0.5*b/g )

# [a] generate a couple of Wiener processes

t_fin = 10.0
dt = 0.001
N_sam = math.ceil(t_fin/dt)
wp1 = np.sqrt(dt) * standard_wiener_process(N_sam)
wp2 = np.sqrt(dt) * standard_wiener_process(N_sam)
wp3 = np.sqrt(dt) * standard_wiener_process(N_sam)
t_vec = np.arange(0, t_fin, dt)

# [b] estimate velocity variance

gamma = np.array([0.3, 1.0])
Lambda = np.array([1.0, 2.0])

alpha_0 = -dt*gamma+1
alpha_1 = np.sqrt( dt*Lambda )

M_avg = 2500

# INITIAL VELOCITY
v_0 = 1.0

v1_matrix = np.zeros((N_sam, M_avg), dtype=np.double)
v2_matrix = np.zeros((N_sam, M_avg), dtype=np.double)

v1_matrix[ 0, : ] = v_0
v2_matrix[ 0, : ] = v_0
for k in range(0, N_sam-1) :
    v1_matrix[ k+1, : ] = alpha_0[0]*v1_matrix[ k, : ] \
        + alpha_1[0]*np.random.normal(0.0, 1.0, M_avg)
    v2_matrix[ k+1, : ] = alpha_0[1]*v2_matrix[ k, : ] \
        + alpha_1[1]*np.random.normal(0.0, 1.0, M_avg)

v1_mean = v1_matrix.mean(axis=1)
v2_mean = v2_matrix.mean(axis=1)

v1_diff2 = np.power( ( v1_matrix.transpose() - \
    v1_mean ).transpose(), 2 )
v2_diff2 = np.power( ( v2_matrix.transpose() - \
    v2_mean ).transpose(), 2 )

v1_var = v1_diff2.mean(axis=1)
v2_var = v2_diff2.mean(axis=1)

# [d] estimate position variance

x1_matrix = np.zeros((N_sam, M_avg), dtype=np.double)
x2_matrix = np.zeros((N_sam, M_avg), dtype=np.double)
for k in range(0, N_sam-1) :
    x1_matrix[ k+1, : ] = x1_matrix[ k, : ] + dt*v1_matrix[ k, : ]
    x2_matrix[ k+1, : ] = x2_matrix[ k, : ] + dt*v2_matrix[ k, : ]

x1_mean = x1_matrix.mean(axis=1)
x2_mean = x2_matrix.mean(axis=1)

x1_diff2 = np.power( ( x1_matrix.transpose() - x1_mean ).transpose(), 2 )
x2_diff2 = np.power( ( x2_matrix.transpose() - x2_mean ).transpose(), 2 )

x1_var = x1_diff2.mean(axis=1)
x2_var = x2_diff2.mean(axis=1)

# Plotting

plt.figure(1)
plt.plot(t_vec, wp1, 'k-')
plt.plot(t_vec, wp2, 'b-')
plt.plot(t_vec, wp3, 'r-')
plt.xlim([0,t_fin])
plt.xlabel('t', fontsize=20)
plt.ylabel('w(t)', fontsize=20)
plt.title('Wiener process realization', fontsize=20)
plt.show()

plt.figure(2)
plt.plot(t_vec, v1_mean, 'b-', label='$\gamma$ = 0.3, $\Gamma$ = 1.0')
plt.plot(t_vec, v2_mean, 'r-', label='$\gamma$ = 1.0, $\Gamma$ = 2.0')
plt.plot(t_vec, v_0*np.exp(-gamma[0]*t_vec), 'b--')
plt.plot(t_vec, v_0*np.exp(-gamma[1]*t_vec), 'r--')
plt.xlim([0,t_fin])
plt.xlabel('t', fontsize=20)
plt.ylabel('<v(t)>', fontsize=20)
plt.title('Langevin equation (velocity)', fontsize=20)
plt.legend(prop=dict(size=18))
plt.show()

plt.figure(3)
plt.plot(t_vec, v1_var, 'b-', label='$\gamma$ = 0.3, $\Gamma$ = 1.0')
plt.plot(t_vec, v2_var, 'r-', label='$\gamma$ = 1.0, $\Gamma$ = 2.0')
plt.plot(t_vec, 0.5*(Lambda[0]/gamma[0]) * ( -np.exp(-2*gamma[0]*t_vec) + 1), 'b--')
plt.plot(t_vec, 0.5*(Lambda[1]/gamma[1]) * ( -np.exp(-2*gamma[1]*t_vec) + 1), 'r--')
plt.xlim([0,t_fin])
plt.xlabel('t', fontsize=20)
plt.ylabel('Var[v(t)]', fontsize=20)
plt.title('Langevin equation (velocity)', fontsize=20)
plt.legend(prop=dict(size=18))
plt.show()

plt.figure(4)
plt.plot(t_vec, x1_mean, 'b-', label='$\gamma$ = 0.3, $\Gamma$ = 1.0')
plt.plot(t_vec, x2_mean, 'r-', label='$\gamma$ = 1.0, $\Gamma$ = 2.0')
plt.plot(t_vec, (v_0/gamma[0]) * ( -np.exp(-gamma[0]*t_vec) + 1 ), 'b--')
plt.plot(t_vec, (v_0/gamma[1]) * ( -np.exp(-gamma[1]*t_vec) + 1 ), 'r--')
plt.xlim([0,t_fin])
plt.xlabel('t', fontsize=20)
plt.ylabel('<x(t)>', fontsize=20)
plt.title('Langevin equation (position)', fontsize=20)
plt.legend(prop=dict(size=18))
plt.show()

plt.figure(5)
plt.plot(t_vec, x1_var, 'b-', label='$\gamma$ = 0.3, $\Gamma$ = 1.0')
plt.plot(t_vec, x2_var, 'r-', label='$\gamma$ = 1.0, $\Gamma$ = 2.0')
plt.plot(t_vec, variance_position_langevin(gamma[0], Lambda[0], t_vec), 'b--')
plt.plot(t_vec, variance_position_langevin(gamma[1], Lambda[1], t_vec), 'r--')
plt.xlim([0,t_fin])
plt.xlabel('t', fontsize=20)
plt.ylabel('Var[x(t)]', fontsize=20)
plt.title('Langevin equation (position)', fontsize=20)
plt.legend(prop=dict(size=18))
plt.show()
