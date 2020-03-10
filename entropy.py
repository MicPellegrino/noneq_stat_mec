from math import sqrt
import numpy as np
import numpy.linalg as la
import numpy.random as rng
import matplotlib.pyplot as plt

def kullback_leibler( x ) :
    return -np.sum(x*np.log(x))

def complex_random_vector ( n ) :
    return np.exp(1j*rng.uniform(0.0,2*np.pi,n))*rng.uniform(0.0,1.0,n)

def random_state( n ):
    psi = complex_random_vector(n)
    return psi/la.norm(psi)

def singular_values( psi ):
    n = len(psi)
    P = np.reshape(psi, ( int(sqrt(n)), int(sqrt(n)) ))
    _, D, _ = la.svd(P)
    return D

def entropy( psi ) :
    lam = singular_values(psi)
    sig = lam*lam
    return kullback_leibler(sig)

# TEST
# N = 12
# n = N*N
# rv = complex_random_vector(n)
# psi = random_state(n)
# t = np.linspace(0.0,2.0*np.pi,250)
# D = singular_values( psi )
# plt.plot(rv.real, rv.imag, 'kx')
# plt.plot(psi.real, psi.imag, 'b.')
# plt.plot(np.cos(t), np.sin(t), 'r-')
# plt.axis('equal')
# plt.show()
# v = np.ones(n)
# psi = v/la.norm(v)
# P = np.reshape(psi, ( int(sqrt(n)), int(sqrt(n)) ))
# _, D, _ = la.svd(P)
# print(D)
# print(entropy(psi))

N_vec = 2*np.array(range(1,7), dtype=int)
M = 5000
S = np.zeros((M, len(N_vec)), dtype=float)
for i in range(len(N_vec)) :
    N = N_vec[i]
    n = int(2**N)
    for j in range(M) :
        psi = random_state(n)
        S[j,i] = entropy(psi)
S_mean = np.mean(S, axis=0)
S_std = np.std(S,axis=0)
print(S_mean)
print(S_std)

plt.plot(N_vec, 0.5*N_vec, 'b-', label='Theoretical limit')
# plt.errorbar(N_vec, S_mean, yerr=S_std, fmt='x', color='b', ls='none', capsize=5, label='Mean')
plt.boxplot(S, positions=N_vec)
# plt.plot(N_vec, S.transpose(), 'k.', markersize=0.25)
plt.legend(loc='upper left', fontsize=20.0)
plt.xlabel('N', fontsize=20.0)
plt.ylabel('Entropy', fontsize=20.0)
plt.tick_params(axis="x", labelsize=20.0)
plt.tick_params(axis="y", labelsize=20.0)
plt.show()
