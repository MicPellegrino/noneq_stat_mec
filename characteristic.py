import numpy as np

import matplotlib.pyplot as plt

N = 100.0
r = 100.0
g = np.array([100, 200, 500])

f = lambda t : ( N/(r+1) ) * ( 1 + r * np.exp(-np.outer(g,t)) )
t_max = 7.0/np.min(g)

t = np.linspace(0, t_max, 250)
n = f(t)

plt.plot(t, n[0,:], linewidth=1.5, label='gamma='+str(g[0]))
plt.plot(t, n[1,:], linewidth=1.5, label='gamma='+str(g[1]))
plt.plot(t, n[2,:], linewidth=1.5, label='gamma='+str(g[2]))
plt.xlim([0,t_max])
plt.ylim([0,1.5*N])
plt.legend(fontsize=20)
plt.xlabel('t', fontsize=20)
plt.ylabel('<n(t)>', fontsize=20)
plt.title('Expected number of molecules A for r='+str(r), fontsize=20)
plt.show()
