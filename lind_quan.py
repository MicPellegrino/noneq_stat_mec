import numpy as np
import matplotlib.pyplot as plt

q_re_rho = [[],[]]
q_im_rho = [[],[]]
l_re_rho = [[],[]]
l_im_rho = [[],[]]

q_re_rho[0].append( np.empty(0, dtype=float) )
q_re_rho[0].append( np.empty(0, dtype=float) )
q_re_rho[1].append( np.empty(0, dtype=float) )
q_re_rho[1].append( np.empty(0, dtype=float) )
q_im_rho[0].append( np.empty(0, dtype=float) )
q_im_rho[0].append( np.empty(0, dtype=float) )
q_im_rho[1].append( np.empty(0, dtype=float) )
q_im_rho[1].append( np.empty(0, dtype=float) )

l_re_rho[0].append( np.empty(0, dtype=float) )
l_re_rho[0].append( np.empty(0, dtype=float) )
l_re_rho[1].append( np.empty(0, dtype=float) )
l_re_rho[1].append( np.empty(0, dtype=float) )
l_im_rho[0].append( np.empty(0, dtype=float) )
l_im_rho[0].append( np.empty(0, dtype=float) )
l_im_rho[1].append( np.empty(0, dtype=float) )
l_im_rho[1].append( np.empty(0, dtype=float) )

tmp = []
with open('Quan/re_rho_00.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    q_re_rho[0][0] = np.array( tmp )
tmp = []
with open('Quan/re_rho_01.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    q_re_rho[0][1] = np.array( tmp )
tmp = []
with open('Quan/re_rho_10.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    q_re_rho[1][0] = np.array( tmp )
tmp = []
with open('Quan/re_rho_11.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    q_re_rho[1][1] = np.array( tmp )
tmp = []
with open('Quan/im_rho_01.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    q_im_rho[0][1] = np.array( tmp )
tmp = []
with open('Quan/im_rho_10.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    q_im_rho[1][0] = np.array( tmp )

tmp = []
with open('Lind/re_rho_00.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    l_re_rho[0][0] = np.array( tmp )
tmp = []
with open('Lind/re_rho_01.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    l_re_rho[0][1] = np.array( tmp )
tmp = []
with open('Lind/re_rho_10.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    l_re_rho[1][0] = np.array( tmp )
tmp = []
with open('Lind/re_rho_11.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    l_re_rho[1][1] = np.array( tmp )
tmp = []
with open('Lind/im_rho_01.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    l_im_rho[0][1] = np.array( tmp )
tmp = []
with open('Lind/im_rho_10.txt', 'r') as f:
    for line in f:
        tmp.append(float(line.split()[0]))
    l_im_rho[1][0] = np.array( tmp )

dt = 0.002
t_fin = 0.5
K = int(t_fin/dt)+1
t = np.linspace(0,t_fin,K)

plt.plot(t, q_re_rho[0][0] ,'r-', label=r'$\rho_{ee}$')
plt.plot(t, q_re_rho[1][1] ,'b-', label=r'$\rho_{gg}$')
plt.plot(t, q_re_rho[0][1] ,'k-', label=r'Re[$\rho_{eg}$]')
plt.plot(t, q_re_rho[1][0] ,'g-', label=r'Re[$\rho_{ge}$]')
plt.plot(t, q_im_rho[0][1] ,'c-', label=r'Im[$\rho_{eg}$]')
plt.plot(t, q_im_rho[1][0] ,'m-', label=r'Im[$\rho_{ge}$]')
plt.plot(t, l_re_rho[0][0] ,'r--')
plt.plot(t, l_re_rho[1][1] ,'b--')
plt.plot(t, l_re_rho[0][1] ,'k--')
plt.plot(t, l_re_rho[1][0] ,'g--')
plt.plot(t, l_im_rho[0][1] ,'c--')
plt.plot(t, l_im_rho[1][0] ,'m--')
plt.legend(fontsize=20.0)
plt.xlim([0,t_fin])
plt.xlabel(r'$t$', fontsize=20.0)
plt.ylabel(r'$\rho(t)$', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.title(r'$\omega$=3, $\Gamma$=1, $n$=10', fontsize=20.0)
plt.show()
