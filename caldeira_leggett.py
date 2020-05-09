import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
from matplotlib import cm

# TEST: velocity Verlet for harmonic oscillator

"""

def velocity_verlet(x_old, v_old, acc, dt):
    a_old = acc(x_old)
    x_new = x_old + dt*v_old + 0.5*dt*dt*a_old
    a_new = acc(x_new)
    v_new = v_old + 0.5*dt*( a_old + a_new )
    return x_new, v_new

omega = 1.0
m = 1.0
acc = lambda x : -(omega*omega/m)*x

x_0 = 0.0
v_0 = 1.0

dt = 0.1
T = 10

N = int(T/dt)
x = np.empty(N+1)
x[0] = x_0
v = np.empty(N+1)
v[0] = v_0

for k in range(N) :
    x[k+1], v[k+1] = velocity_verlet(x[k], v[k], acc, dt)

"""

"""
    Set of parameters to test:
    [1]
        cutoff = 1.0
        alpha = 1.0
        N = 10, 100, 500
        dt = 0.1
        T = 100.0
    [2]
        cutoff = 1.0, 10.0, 100.0
        alpha = 1.0
        N = 1000
        dt = 0.005
        T = 25.0
"""

def spectral_density(freq, coeff, cutoff):
    return freq*coeff*(freq<cutoff)

high_cut_off = 1.0
alpha = 1.0

SD = lambda w : spectral_density(w, alpha, high_cut_off)

# Number of independent oscillator
N = 10

dw = high_cut_off/N
w = np.zeros(N+1)
# Staggered frequencies
# w[1:] = np.linspace(0.0, high_cut_off-dw, N) + 0.5*dw
# Nodal frequencies
w[1:] = np.linspace(dw, high_cut_off, N)
t_rec = 2*np.pi/dw

# TEST
# print('Discrete frequencies values:')
# print(w)
print('Estimate recurrence time:')
print(t_rec)

c = np.zeros(N+1)
# Explicit Ohmic sd
# c[1:] = np.sqrt(alpha*dw*2.0/np.pi) * w[1:]
# Generic sd
c[1:] = np.sqrt( 2.0*w[1:]*dw*SD(w[1:])/np.pi )
# With counter term
# c[0] = -2*alpha*high_cut_off/np.pi
c[0] = - np.sum( (c[1:]**2) / (w[1:]**2) )
# Without counter term
# c[0] = 0.0
# print(c)

m = np.ones(N+1)
m[0] = 0.1
acc_0 = lambda x : np.inner(c, x)/m[0]
acc_k = lambda x, k : c[k]*x[0] - (w[k]**2)*x[k]
def acc(x) :
    a = np.zeros(N+1)
    a[0] = acc_0(x)
    for k in range(1,N+1) :
        a[k] = acc_k(x,k)
    return a

# Lambda expressions for kinetic, potential and energy
kin_ener = lambda v : 0.5 * np.sum( m*v**2 )
pot_ener = lambda x : 0.5 * np.sum( ( w[1:]*x[1:] - (c[1:]/w[1:])*x[0] )**2 )
tot_ener = lambda x, v : kin_ener(v) + pot_ener(x)

# Initial positions (zeros)
x_init = np.zeros(N+1)
v_init = np.zeros(N+1)

# Initial velocities given initial kinetic energy
beta = 1.0
K_s = 0.5*N/beta
v_init = rng.normal(0.0, 1.0, N+1)
# v_init[1:] = np.sqrt( K_s / kin_ener(v_init[1:]) ) * v_init[1:]
# v_init[0] = 0.0
v_init[1:] = np.sqrt( 2.0 * K_s / np.sum( v_init[1:]**2 ) ) * v_init[1:]
v_init[0] = 1.0

def velocity_verlet(x_old, v_old, acc, dt):
    a_old = acc(x_old)
    x_new = x_old + dt*v_old + 0.5*dt*dt*a_old
    a_new = acc(x_new)
    v_new = v_old + 0.5*dt*( a_old + a_new )
    return x_new, v_new

dt = 0.1

# Ensure stability for the numerical algorithm
recall_freq = max( np.sqrt(-c[0]), high_cut_off )
print('Max. recall frequency:')
print(recall_freq)
assert dt < 2.0 / recall_freq, "Time step is too large!"

T = 100
M = int(T/dt)-1
t_vec = np.linspace(0,T,M+1)

# Number of thermostat steps
M_thermo = 10

x_old = x_init
v_old = v_init
x_par = []
x_par.append(x_old[0])
v_par = []
v_par.append(v_old[0])

# Energy vectors
K = []
K.append( kin_ener(v_old) )
P = []
P.append( pot_ener(x_old) )
H = []
H.append( tot_ener(x_old, v_old) )

# TEST
print('Initial total energy:')
print(H[0])

for n in range(M) :
    x_new, v_new = velocity_verlet(x_old, v_old, acc, dt)
    x_par.append(x_new[0])
    v_par.append(v_new[0])
    K.append( kin_ener(v_new) )
    P.append( pot_ener(x_new) )
    H.append( tot_ener(x_new, v_new) )
    # if n % M_thermo == 0 :
    #     v_new[1:] = np.sqrt( K_s / kin_ener(v_new[1:]) ) * v_new[1:]
    x_old = x_new
    v_old = v_new

x_par = np.array(x_par)
# print(x_par)
v_par = np.array(v_par)
# print(v_par)

# Construct recurrence map
eps = 0.02
"""
R = np.zeros((M+1,M+1), dtype=int)
for i in range(M+1) :
    for j in range(M+1) :
        R[i,j] = np.abs(x_par[i]-x_par[j]) <= eps
"""
Xi = np.outer( x_par, np.ones(M+1) )
Xj = np.outer( np.ones(M+1), x_par )
diff = np.abs( Xi-Xj )
R = (diff<=eps).astype(int)

plt.plot(t_vec, H, 'k-', label='total')
plt.plot(t_vec, K, 'r-', label='kinetic')
plt.plot(t_vec, P, 'b-', label='potential')
plt.xlabel('time')
plt.ylabel('energy')
# plt.ylim([0, 3.0*H[0]])
plt.legend()
plt.show()

plt.plot(x_par, v_par, 'k-')
plt.xlabel('position')
plt.ylabel('velocity')
plt.show()

plt.plot(t_vec, x_par, 'k-')
plt.xlabel('time')
plt.ylabel('position')
plt.show()

"""
plt.matshow(R, cmap=cm.binary)
# plt.axis('equal')
plt.show()
"""
