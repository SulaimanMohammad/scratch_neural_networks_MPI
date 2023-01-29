import NNModel as nn
import NNModelPara as npar
import timeit
import numpy as np
from math import pi, sqrt
from time import sleep
import matplotlib.pyplot as plt 
import bigfloat as bf
bf.setcontext(bf.precision(200))


#------------------------------------------------------------------------------------#
#|||||||||||||||||||||||open models and prepare data ||||||||||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#
#network_wights=nn.build_network(inputs,outputs,x,3, 0.2,1000, "sigmoid", history=True)

network_wights1 =nn.open_model("Xor10*3")
network_wights2 =nn.open_model("Xor30*3")
network_wights3 =nn.open_model("Xor50*3")
network_wights4 =nn.open_model("Xor60*3")

network=[network_wights1,network_wights2,network_wights3,network_wights4]
neurons=[10,30,50,60]
error_relative=list()
times=list()
times_para=list()
error_relative_para=list()
error_relative_para_mpfr=list()
err=0


#------------------------------------------------------------------------------------#
#|||||||||||||||||||Test full network Seq, Parallel, MPFR |||||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#
for net in network:
    
    #------------------------------Seq--------------------------------------#
    n=1000
    time_ex=0
    for i in range(n):
        starttime = timeit.default_timer()
        res=nn.predict(net,[1, 0, 1],"sigmoid")
        time_ex += timeit.default_timer() - starttime
    time_ex=time_ex/n
    times.append(time_ex)
    err=bf.abs(bf.div( bf.sub(res[0],1) ,1) )
    error_relative.append(err)
    
    #------------------------------Parallel----------------------------------#
    time_ex=0
    for i in range(n):
        res=npar.predict_parallel(net,[1, 0, 1], 'sigmoid', time_mesur=True, print_res=True) 
        if res is not None:
            time_ex +=res[1]
            result=res[0]
    time_ex=time_ex/n
    if res is not None:
        times_para.append(time_ex)
        res=res[0]
        err=bf.abs(bf.div( bf.sub(float(res[0]),1) ,1) )
        error_relative_para.append(err)

    #---------------------------Parallel_MPFR--------------------------------#
    n=1
    time_ex=0
    for i in range(n):
        res=npar.predict_parallel_mpfr(net,[1, 0, 1], 200, 'sigmoid', time_mesur=True, print_res=True) 
    if res is not None:
        res=res[0]
        err=bf.abs(bf.div( bf.sub(res[0],1) ,1) )
        error_relative_para_mpfr.append(err)

    #----------------------------Plot data----------------------------------#

if res is not None:# the process 0 is the only one return value so we plot just one time

    plt.subplot(3,2,1)
    plt.plot (neurons, times, marker='o',color='black', linestyle='dashed', label="serial")
    plt.plot (neurons, times_para, marker='x',color='red', linestyle='dashed', label="parallel")
    
    plt.title("8 processes")
    plt.yscale("log")
    plt.xlabel("neuron")
    plt.ylabel("time")
    plt.legend()

    plt.subplot(3,2,2)
    plt.plot (neurons, error_relative, marker='o',color='black', linestyle='dashed', label="serial")
    plt.plot (neurons, error_relative_para, marker='x',color='red', linestyle='dashed', label="parallel")
    plt.plot (neurons, error_relative_para_mpfr, marker='D',color='green', linestyle='dashed', label="parallel_mpf4") 
    plt.legend()




#------------------------------------------------------------------------------------#
#||||||||||||||Test pruning connection network Seq, Parallel ||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#

net1= network_wights1.copy() # remeber in python the lists are passed by reference so they changed for that i do copy of the original 
net2= network_wights2.copy() 
net3= network_wights3.copy() 
net4= network_wights4.copy() 
net1= nn.pruning_connections( net1, 0.4)
net2= nn.pruning_connections( net2, 0.4)
net3= nn.pruning_connections( net3, 0.4)
net4= nn.pruning_connections( net4, 0.4)

network=[net1,net2,net3,net4]
error_relative=list()
times=list()
times_para=list()
error_relative_para=list()


for netw in network:

    #------------------------------Seq--------------------------------------#
    n=1000
    time_ex=0
    for i in range(n):
        starttime = timeit.default_timer()
        res=nn.predict(netw,[1, 0, 1],"sigmoid")
        time_ex += timeit.default_timer() - starttime
    time_ex=time_ex/n
    times.append(time_ex)
    err=bf.abs(bf.div( bf.sub(res[0],1) ,1) )
    error_relative.append(err)
    
    #------------------------------Parallel----------------------------------#
    time_ex=0
    for i in range(n):
        res=npar.predict_parallel(netw,[1, 0, 1], 'sigmoid', time_mesur=True, print_res=True) 
        if res is not None:
            time_ex +=res[1]
            result=res[0]
    time_ex=time_ex/n
    if res is not None:
        times_para.append(time_ex)
        res=res[0]
        err=bf.abs(bf.div( bf.sub(res[0],1) ,1) )
        error_relative_para.append(err)

    #----------------------------Plot data----------------------------------#
if res is not None:# the process 0 is the only one return value so we plot just one time
    plt.subplot(3,2,3)
    plt.plot (neurons, times, marker='+',color='yellow', linestyle='dashed', label="seq_prun")
    plt.plot (neurons, times_para, marker='D',color='green', linestyle='dashed', label="parallel_prun")
    plt.yscale("log")
    plt.xlabel("neuron")
    plt.ylabel("time")
    plt.legend()

    plt.subplot(3,2,4)
    plt.plot (neurons, error_relative, marker='o',color='black', linestyle='dashed', label="serial")
    plt.plot (neurons, error_relative_para, marker='x',color='red', linestyle='dashed', label="parallel")
    plt.legend()




#------------------------------------------------------------------------------------#
#||||||||||||||Test pruning neuron network Seq, Parallel ||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#
neu1= network_wights1.copy() # remeber in python the lists are passed by reference so they changed for that i do copy of the original 
neu2= network_wights2.copy() 
neu3= network_wights3.copy() 
neu4= network_wights4.copy() 
neu1= nn.pruning_neuron( neu1, 0.3)
neu2= nn.pruning_neuron( neu2, 0.3)
neu3= nn.pruning_neuron( neu3, 0.3)
neu4= nn.pruning_neuron( neu4, 0.3)

network=[neu1,neu2,neu3,neu4]
error_relative=list()
times=list()
times_para=list()
error_relative_para=list()

for netw in network:

    #------------------------------Seq--------------------------------------#
    n=1000
    time_ex=0
    for i in range(n):
        starttime = timeit.default_timer()
        res=nn.predict(netw,[1, 0, 1],"sigmoid")
        time_ex += timeit.default_timer() - starttime
    time_ex=time_ex/n
    times.append(time_ex)
    err=bf.abs(bf.div( bf.sub(res[0],1) ,1) )
    error_relative.append(err)

    #------------------------------Parallel----------------------------------#
    time_ex=0
    for i in range(n):
        res=npar.predict_parallel(netw,[1, 0, 1], 'sigmoid', time_mesur=True, print_res=True) 
        if res is not None:
            time_ex +=res[1]
            result=res[0]
    time_ex=time_ex/n
    if res is not None:
        times_para.append(time_ex)
        res=res[0]
        err=bf.abs(bf.div( bf.sub(res[0],1) ,1) )
        error_relative_para.append(err)

    #----------------------------Plot data----------------------------------#
if res is not None:# the process 0 is the only one return value so we plot just one time
    plt.subplot(3,2,5)
    plt.plot (neurons, times, marker='+',color='yellow', linestyle='dashed', label="seq_prun")
    plt.plot (neurons, times_para, marker='D',color='green', linestyle='dashed', label="parallel_prun")
    plt.yscale("log")
    plt.xlabel("neuron")
    plt.ylabel("time")
    plt.legend()

    plt.subplot(3,2,6)
    plt.plot (neurons, error_relative, marker='o',color='black', linestyle='dashed', label="serial")
    plt.plot (neurons, error_relative_para, marker='x',color='red', linestyle='dashed', label="parallel")
    
    plt.xlabel("neuron")
    plt.ylabel("error_relative")
    plt.yscale("log")
    plt.legend()



plt.show() 
