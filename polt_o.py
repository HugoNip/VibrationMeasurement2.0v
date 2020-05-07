# load numpy array from csv file
from numpy import loadtxt
import matplotlib.pyplot as plt

# load array
'''
data_C_Opt = loadtxt('data_O_Circle.csv', delimiter=',')
data_C_Cir = loadtxt('data_O_C.csv', delimiter=',')
'''

data_C_Opt = loadtxt('data_O_Square_45.csv', delimiter=',')
data_C_Cir = loadtxt('data_I_S_45.csv', delimiter=',')

# print(data_C_Cir)
# print(data_C_Opt)

plt.plot(data_C_Opt, 'r--', label = 'Optical Flow',)
plt.plot(data_C_Cir, 'b', label = 'Intersection',)
plt.axis([0, 500, -10, 70])
plt.ylabel('Displcement (pixels)')
plt.xlabel('Time(sec)')
plt.title('Vertical Displacement')
plt.savefig('Square_vDisp.png')
plt.show()