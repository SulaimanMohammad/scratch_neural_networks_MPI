# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import timeit
import NNModel as nn

# load the dataset
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0], [0], [0], [1], [1], [1]])

# define the keras model
model = Sequential()
model.add(Dense(60, input_dim=3, activation='sigmoid'))
model.add(Dense(60, activation='sigmoid'))
model.add(Dense(60, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(inputs, outputs, epochs=1000)
print("\n-------------------------------------------------------\n")




# evaluate the keras model
_, accuracy = model.evaluate(inputs, outputs)
print('Accuracy: %.2f' % (accuracy*100))

starttime = timeit.default_timer()

res=model.predict([[1, 0, 1] ] ) 

time_ex= timeit.default_timer() - starttime
print("tenser flow      ","\t res", res, "\t time:",time_ex )
print("-------------------------------------------------------")

network_wights =nn.open_model("Xor60*3")
n=1000
time_ex=0
for i in range(n):
    starttime = timeit.default_timer()
    res=nn.predict(network_wights,[1, 0, 1],"sigmoid")
    time_ex += timeit.default_timer() - starttime
time_ex=time_ex/n
print("My network      ","\tres:",res,"\t time:",time_ex )