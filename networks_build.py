import NNModel2 as nn
import NNModelPara as npar
import NNModelMpfr as nmp
import timeit
import numpy as np
from math import pi, sqrt
from time import sleep

inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0], [0], [0], [1], [1], [1]])


# ------------ ------------------------------------------
# this file to build networks will be opened in implantation_para
#--------------------------------------------------------
network_wights=nn.build_network(inputs,outputs,60,3, 0.2,1000, "sigmoid", history=True)

nn.save_model(network_wights, 'Xor60*3')

n=1
time_ex=0
for i in range(n):
    starttime = timeit.default_timer()
    res=nn.predict(network_wights,[1, 0, 1],"sigmoid")
    time_ex += timeit.default_timer() - starttime
time_ex=time_ex/n

print("normal                    ", res, "\t time", time_ex)
