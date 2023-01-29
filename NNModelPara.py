import numpy as np
from numpy import array
from mpi4py import MPI
from math import exp
from operator import add
import pickle



# Transfer neuron activation
def activations(product_result,func_type):
    if func_type=="Sigmoid" or func_type=="sigmoid": return 1.0 / (1.0 + exp(-product_result))
    if func_type== "TanH" or func_type=="tanh": return (2 / (1.0 + exp(-2*product_result)))-1
    if func_type== "Linear" or func_type=="linear": return product_result  
    if func_type== "Relu" or func_type=="relu": return max(0.0,product_result)
    if func_type== "LRelu" or func_type=="lrelu": return np.where(product_result > 0.0, product_result, product_result* 0.01) 
       

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_comm=comm.Get_size()


# here we just take input * wights and sigmoid for each node there is no need to have the output of others 
def process_prod(network, row,func_type ):
        activation1 =[]
        inputs = row
        for neuron, i in zip(network, range( len(network)  )): #iterate in the inut and the neurons 
            calc=0
            activation=[] # intermidate list to append all the multiple , then after will put it in activation1
            weights=neuron['weights']
            # for j in range(len(weights)-1):
            #     calc+=weights[j] * inputs[j]
            # calc+=weights[-1]
            data= weights[: len(weights)-1]
            data=np.array(data)
            data=data.astype(float)
            inputs=np.array(inputs)
            inputs=inputs.astype(float)
            calc=np.dot(data,inputs )+weights[-1]
            calc=activations(calc,func_type)
            activation1.append(calc)
        return activation1  



def predict_parallel(network_wights,inputs, func_type='sigmoid', time_mesur=False , print_res=False):

    buf = np.zeros(1)
    if rank==0: 
        buf[0] = MPI.Wtime()

    for net_par in network_wights:
        chunk=int(len(net_par)/size_comm)
        #check how to do for 5 nurons but 2 process , so imposibble to divide
        if(len(net_par) % size_comm ==0):
            data =process_prod(net_par[rank*chunk: (rank*chunk)+chunk], inputs, func_type) # this will be performed by all no need for if rank==0 it works for all 
        else:
            if rank==size_comm-1:
                data =process_prod(net_par[rank*chunk:], inputs, func_type) # this will be performed by all no need for if rank==0 it works for all                
            else:
                data =process_prod(net_par[rank*chunk: (rank*chunk)+chunk], inputs, func_type) # this will be performed by all no need for if rank==0 it works for all 

        inputs=comm.gather(data,root=0) # gather all to have one input 

        if rank==0:
            new_inputs_forall = []
            for sublist in inputs:
                for item in sublist:
                    new_inputs_forall.append(item)
            inputs=new_inputs_forall
        else:
            inputs=None
            
        inputs=comm.bcast(inputs, root=0) # now the input after the first layer is for all 
    
    if rank==0:
        if time_mesur==True and print_res==True:
            after_timestamp = MPI.Wtime()
            elapsed = after_timestamp - buf[0]
            #print(f"It took {elapsed} ")
        return [inputs, elapsed]
# we will have the right reslt where ther is no data from others 




import bigfloat as bf
def predict_parallel_mpfr(network_wights,inputs,precision, func_type='sigmoid', time_mesur=False , print_res=False):
    bf.setcontext(bf.precision(precision))  
    def activation_mpfr(product_result,func_type):       
        if func_type=="Sigmoid" or func_type=="sigmoid": return np.float64 (bf.div(1,  bf.add(1,bf.exp(-product_result)))) #1.0 / (1.0 + exp(-product_result))
        if func_type== "TanH" or func_type=="tanh": return np.float64( bf.div(1,bf.add(1,bf.exp( bf.mul(-2,product_result)))) )#(2 / (1.0 + exp(-2*product_result)))-1

    # here we just take input * wights and sigmoid for each node there is no need to have the output of others 
    def process_prod_mpfr(network, row,func_type ):
            activation1 =[]
            inputs = row
            for neuron, i in zip(network, range( len(network)  )): #iterate in the inut and the neurons 
                calc=0
                activation=[] # intermidate list to append all the multiple , then after will put it in activation1
                weights=neuron['weights']
                for j in range(len(weights)-1):
                    calc= bf.add(calc, bf.mul(weights[j] , inputs[j]) )
                calc=bf.add(calc, weights[-1])
                calc=activation_mpfr(calc,func_type)
                activation1.append(calc)
            return activation1 

    buf = np.zeros(1)
    if rank==0: 
        buf[0] = MPI.Wtime()

    for net_par in network_wights:
        chunk=int(len(net_par)/size_comm)
        #check how to do for 5 nurons but 2 process , so imposibble to divide
        if(len(net_par) % size_comm ==0):
            data =process_prod_mpfr(net_par[rank*chunk: (rank*chunk)+chunk], inputs, func_type) # this will be performed by all no need for if rank==0 it works for all 
        else:
            if rank==size_comm-1:
                data =process_prod_mpfr(net_par[rank*chunk:], inputs, func_type) # this will be performed by all no need for if rank==0 it works for all                
            else:
                data =process_prod_mpfr(net_par[rank*chunk: (rank*chunk)+chunk], inputs, func_type) # this will be performed by all no need for if rank==0 it works for all 
        
        inputs=comm.gather(data,root=0) # gather all to have one input 

        if rank==0:
            new_inputs_forall = []
            for sublist in inputs:
                for item in sublist:
                    new_inputs_forall.append(item)
            inputs=new_inputs_forall
        else:
            inputs=None
            
        inputs=comm.bcast(inputs, root=0) # now the input after the first layer is for all 
    
    if rank==0:
        if time_mesur==True and print_res==True:
            after_timestamp = MPI.Wtime()
            elapsed = after_timestamp - buf[0]
            #print(f"It took {elapsed} ")
        return [inputs, elapsed]


