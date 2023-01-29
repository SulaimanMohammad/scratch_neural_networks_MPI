import numpy as np # helps with the math
from math import exp
from math import log
from random import seed
from random import random
from numpy import array, product
import pickle



# Transfer neuron activation
def activation(product_result,func_type):
    if func_type=="Sigmoid" or func_type=="sigmoid": return 1.0 / (1.0 + exp(-product_result))
    if func_type== "TanH" or func_type=="tanh": return (2 / (1.0 + exp(-2*product_result)))-1
    if func_type== "Linear" or func_type=="linear": return product_result  
    if func_type== "Relu" or func_type=="relu": return max(0.0,product_result)
    if func_type== "LRelu" or func_type=="lrelu": return np.where(product_result > 0.0, product_result, product_result* 0.01) 
        # if  product_result <0: 
        #     return 0.1*product_result
        # elif  product_result >=0: 
        #     return product_result
        
# Calculate the derivative of an neuron output
def activation_derivative(outputd,func_type):
    if func_type=="Sigmoid" or func_type=="sigmoid": return  outputd * (1.0 - outputd)
    if func_type== "TanH" or func_type=="tanh": return (1- (outputd * outputd) )
    if func_type== "Linear" or func_type=="linear": return 1  
    if func_type== "Relu" or func_type=="relu": 
        if  outputd <0.0: 
            return 0.0
        else: 
            return 1.0 
    if  func_type=="lrelu": 
        if  outputd <0.0: 
            return 0.1
        else: 
            return 1.0 

def initialize_network(inputs,outputs,n_neuron,n_hidden_layers):
        n_inputs=len(inputs[0])
        n_outputs=len(outputs[0])

        #number of each nurons in each layer= input+1  
        # you can see that the wight of each layer depends on the number of what befor and after 
        #W¹ is a weight matrix of shape (n, m) where n is the number of output neurons (neurons in the next layer) and m is the number of input neurons (neurons in the previous layer)
        #since we dont have hemogenuous wights because hidden layer nurons dosnt equal the output so we need dictionary 
        network_wights=[]

        # here the number shoud be depends for so adding another layer should change the dependency 
        #first layer after the input wight should match number of inputs 
        hidden_wights=[]
        for i in range(n_neuron): # for each neroun in the layer  generate input number + baise(1)
            wights=np.random.randn(n_inputs + 1)
            hidden_wights.append({'weights': wights })#append the weights of one neuron
        network_wights.append(hidden_wights)


        # fo the middel after the firt one, depends on the number of the one before 
        for layer in range(n_hidden_layers-1): # beacuse the first one is done 
            hidden_wights=[]
            for i in range(n_neuron): # for each neroun in the layer  generate input number + baise(1)
                wights=np.random.randn(n_neuron + 1)
                hidden_wights.append({'weights': wights }) #append the weights of one neuron
            network_wights.append(hidden_wights)

        hidden_wights=[]
        for i in range(n_outputs): # for each hidden mlayer generate input number + baise(1)
            wights=np.random.randn(n_neuron + 1)
            hidden_wights.append({'weights': wights })
        network_wights.append(hidden_wights)

       
        return network_wights
    
# Calculate neuron activation for an input
#You can see that a neuron’s output value is stored in the neuron with the name ‘output‘. 
#You can also see that we collect the outputs for a layer in an array named new_inputs 
#that becomes the array inputs and is used as inputs for the following layer
#Forward propagate input to a network output
def node_product(weights, inputs):
        data= weights[: len(weights)-1]
        data=np.array(data)
        data=data.astype(float)
        inputs=np.array(inputs)
        inputs=inputs.astype(float)
        product_result=np.dot(data,inputs )+weights[-1]
    # def node_product(weights, inputs):
    #         product_result=0
    #         product_result = weights[-1]
    #         for i in range(len(weights)-1):
    #             product_result += weights[i] * inputs[i]
    #             #print("product_result",product_result)
    #         return product_result
        return product_result

def forward_propagate(network, row , func_type):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                product_result= node_product(neuron['weights'], inputs)
                # print("product_result",product_result,"func_type",func_type )
                neuron['output'] = activation(product_result, func_type)
                new_inputs.append(neuron['output'])
            inputs = new_inputs

        return inputs   

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected, func_type):

        for i in reversed(range(len(network))): #go backwords 
            layer = network[i]
            errors = list()
            if i != len(network)-1: #for the rest of the layers not th output 
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]: # we needs the data of the layer after to use it to back , ( ex for layer 2 we need layer 3)  
                        nw= neuron['weights'][j] 
                        nd=neuron['delta']
                        error += (nw * nd )
                    errors.append(error)

            else:#for the last layer ( final output) 
                for j in range(len(layer)): #find the error between the compute and expected in each neuron for the last layer  
                    neuron = layer[j]
                    errors.append(neuron['output'] - expected[j]) # create list of each neuron respectivly 
            
            for j in range(len(layer)):
                neuron = layer[j]
                grad= activation_derivative(neuron['output'], func_type)
                errj= errors[j]
                neuron['delta'] =  errj * grad #find gradiant of the error 
                          
#
def update_weights(network, row, l_rate):
        for i in range(len(network)):
            inputs = row
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]

            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= (l_rate * neuron['delta'] * inputs[j])
                neuron['weights'][-1] -= l_rate * neuron['delta']
    
#
def train_network(network, train, l_rate, n_epoch, ex_outputs,func_type, history):
        
        for epoch in range(n_epoch): # will train it for the same data set many times until arrive to the final expected data
            sum_error = 0
            for row in range(len(train)): # row will represent each data of the data set 

                outputs = forward_propagate(network, train[row], func_type) # find actual output from the wights we have 
                expected=ex_outputs[row]

                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                backward_propagate_error(network, expected , func_type)
                update_weights(network, train[row], l_rate)

            if(history is True):
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        if sum_error>3:
            print ("\n The error doesn't coverge t0 zero,  \n Change your activation function or the number of epoch \n ")
            if history == False:
                print ("\n To see details of training, Add to your function history = True \n ")
             
global func_type
func_type= "sigmoid"

def build_network(inputs,outputs,n_neuron,n_hidden_layers, l_rate ,n_epoch,fuction_type = "sigmoid", history= True):
    global func_type
    func_type =fuction_type 
    network_wights= initialize_network(inputs,outputs,n_neuron,n_hidden_layers)

    train_network(network_wights, inputs, l_rate, n_epoch, outputs,func_type, history)
    return network_wights


def save_model(network, name):
    with open(name, "wb") as internal_filename:
        pickle.dump(network, internal_filename)

def open_model(name): 
    with open(name, "rb") as new_filename:
        network=pickle.load(new_filename)
    return network


def predict(network, row,func_type):
    outputs = forward_propagate(network, row, func_type)
    return outputs


def predict_mpfr(network, row,func_type,precision):
    import bigfloat as bf 
    bf.setcontext(bf.precision(precision))  
    def activation_mpfr(product_result,func_type):
        bf.setcontext(bf.precision(precision))  
        #bf.setcontext(bf.precision(precision))  
        if func_type=="Sigmoid" or func_type=="sigmoid": return bf.div(1,  bf.add(1,bf.exp(-product_result))) #1.0 / (1.0 + exp(-product_result))
        if func_type== "TanH" or func_type=="tanh": return np.float64( bf.div(2,bf.add(1,bf.exp( bf.mul(-2,product_result))))) #(2 / (1.0 + exp(-2*product_result)))-1

    def node_mpfr(weights, inputs):
        bf.setcontext(bf.precision(precision))  
        product_result = weights[-1]
        for i in (range(len(weights)-1)):
            product_re = bf.mul(weights[i] , inputs[i]) 
            product_result = bf.add(product_result, product_re)
        return product_result

    def forward_propagate_mpfr(network, row , func_type):
        bf.setcontext(bf.precision(precision))  
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                product_result= node_mpfr(neuron['weights'], inputs)
                neuron['output'] = activation_mpfr(product_result, func_type)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs


    outputs = forward_propagate_mpfr(network, row, func_type)


    return outputs


def pruning_neuron(network_wights,ratio):
    nu_neuron=0
    for layer in network_wights:
        for neuron in layer:
            nu_neuron =nu_neuron+1

    n_neuron_purn= int((nu_neuron) *ratio)

    min_list=list() 
    # pruning whole neuron 
    for i, layer in zip(range(len(network_wights)), network_wights):
        for j,neuron in zip(range(len(layer)), layer):
            #dont touch the bias 
            n=len(neuron['weights'])-1 # avoid the bias taking 
            x=neuron['weights'][:n]
            min_loc= sum(abs(x))
            min_list.append( [min_loc, [i,j] ]) # list for each min and the index , then we can sort it 

    min_list= sorted(min_list) # now the sum is sorted from small to large with the index of each one 

    # min_list[:n_neuron_purn] what to prun 
    for neuron in  min_list[:n_neuron_purn]:
        for index in neuron[1:]: # take just the second element where we have the indexes 
            indexs_neuron=index 
            network_wights[indexs_neuron[0]][indexs_neuron[1]]['weights']= np.zeros(len(network_wights[indexs_neuron[0]][indexs_neuron[1]]['weights'])).astype(float)   
    return network_wights

def pruning_connections(network_wights,ratio):
    for layer in network_wights:
        for neuron in layer:
            #dont touch the bias 
            n=len(neuron['weights'])-1 # avoid the bias taking 
            x=neuron['weights'][:n]
            x=abs(x) # find the samllest number regardless the sign, if we kept the sing then we will find always the negative 
            n_purn=int(len(x)*ratio) 
            idx = np.argpartition(x, n_purn) # find the smallest indexs , n_purn elements in the neuron 
            neuron['weights'][idx[:n_purn]]=0.0  # maake them zero 
    return network_wights



#------------------------------------------------------------------------------------#
#|||||||||||||||||||#scipy |||||||||||||||||||||||||||#
#------------------------------------------------------------------------------------#
def node_product_prun(weights, inputs):
        from scipy import sparse
        from scipy import dot 
        data= weights[: len(weights)-1]
        data= sparse.csr_matrix(data)
        product_result=data.dot(inputs)+weights[-1]
        
        return product_result
def forward_propagate_prun(network, row , func_type):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                product_result= node_product_prun(neuron['weights'], inputs)
                # print("product_result",product_result,"func_type",func_type )
                neuron['output'] = activation(product_result, func_type)
                new_inputs.append(neuron['output'])
            inputs = new_inputs

        return inputs   

 