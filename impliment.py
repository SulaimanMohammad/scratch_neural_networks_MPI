import NNModel as nn
import NNModelPara as npar
#import NNModelMpfr as nmp
import timeit
import numpy as np
from math import pi, sqrt
from time import sleep
import bigfloat as bf


#------------------------------------------------------------------------------------#
#|||||||||||||||||||||||open models and prepare data ||||||||||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#
# input data
inputs = ([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = ([[0], [0], [0], [1], [1], [1]])

times1=list()
error_relative1=list()

times2=list()
error_relative2=list()

times3=list()
error_relative3=list()

times_mpfr=list()
error_relative_mpfr=list()

err=0
neuron_numbers=list()
bf.setcontext(bf.precision(200))



for num in range(2,40,8):
    neuron_numbers.append(num*3)

    network_wights=nn.build_network( inputs,outputs,num,3, 0.2 ,int(3000),"sigmoid", history= False)
    
    print("3 layers with number of neuron in each layer",num )

#------------------------------------------------------------------------------------#
#||||||||||||||||||||||||||||Test full network Seq,|||| |||||||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#
    n=1000
    time_ex=0
    for i in range(n):
        starttime = timeit.default_timer()
        res=nn.predict(network_wights,[1, 0, 1],"sigmoid")
        time_ex += timeit.default_timer() - starttime
    time_ex=time_ex/n
    times1.append(time_ex)
    err=bf.abs(bf.div( bf.sub(res[0],1) ,1) )
    error_relative1.append(err)#res[0]-1)/1)
    print("Full network        ","\tres:",res,"\t time:",time_ex )

#------------------------------------------------------------------------------------#
#||||||||||||||||||||||||Test full network Seq,MPFR|||| |||||||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#
    n=1
    time_ex=0
    for i in range(n):
        starttime = timeit.default_timer()
        res=nn.predict_mpfr(network_wights,[1, 0, 1],"sigmoid",200)
        time_ex += timeit.default_timer() - starttime
    time_ex=time_ex/n
    times_mpfr.append(time_ex)
    err=bf.abs(bf.div( bf.sub(res[0],1) ,1) )
    error_relative_mpfr.append(err)#res[0]-1)/1)
    print("Full network,MPfr   ","\tres:",res,"\t time:",time_ex )


#------------------------------------------------------------------------------------#
#|||||||||||||||||||Test pruning_connections network Seq|||||||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#
    neu= network_wights.copy()
    net= network_wights.copy() # remeber in python the lists are passed by reference so they changed for that i do copy of the original 
    net= nn.pruning_connections( net, 0.4)

    n=1000
    time_ex=0
    for i in range(n):
        starttime = timeit.default_timer()
        res=nn.predict(net ,[1, 0, 1],"sigmoid")
        time_ex += timeit.default_timer() - starttime
    time_ex=time_ex/n
    times2.append(time_ex)
    err=bf.abs(bf.div( bf.sub(res[0],1) ,1) )
    error_relative2.append(err)#res[0]-1)/1)
    print("pruning_connections ","\tres:",res,"\t time:",time_ex )



#------------------------------------------------------------------------------------#
#||||||||||||||||||||||||Test pruning_neuron network Seq|||||||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#
    neu=nn.pruning_neuron(neu,0.2)
    time_ex=0
    for i in range(n):
        starttime = timeit.default_timer()
        res=nn.predict(neu ,[1, 0, 1],"sigmoid")
        time_ex += timeit.default_timer() - starttime
    times3.append(time_ex)
    err=bf.abs(bf.div( bf.sub(res[0],1) ,1) )
    error_relative3.append(err)#res[0]-1)/1)
    print("pruning_neuron      ","\tres:",res,"\t time:",time_ex )
    print("-------------------------------------------------------")
 
  
#------------------------------------------------------------------------------------#
#||||||||||||||||||||||||||||||||||||||||||||PLOT||||||||||||||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#
import matplotlib.pyplot as plt 
plt.subplot(1,2,1)

plt.plot (neuron_numbers, times1, marker='o',color="black", linestyle='dashed', label='normal')
plt.plot (neuron_numbers, times2, marker='x',color="red", linestyle='dashed', label='prun_connection')
plt.plot (neuron_numbers, times3, marker='*',color="blue", linestyle='dashed', label='prun_layer')
plt.yscale("log")
plt.xlabel("neuron")
plt.ylabel("time")
plt.legend()

plt.subplot(1,2,2)
plt.plot (neuron_numbers, error_relative1, marker='o',color="black", linestyle='dashed', label='normal')
plt.plot (neuron_numbers, error_relative2, marker='x',color="red", linestyle='dashed', label='prun_connection')
plt.plot (neuron_numbers, error_relative3, marker='*',color="blue", linestyle='dashed', label='prun_layer')
plt.plot (neuron_numbers, error_relative_mpfr, marker='D',color="green", linestyle='dashed', label='normal MPFr')

plt.xlabel("neuron")
plt.ylabel("time")
plt.legend()

plt.show() 
