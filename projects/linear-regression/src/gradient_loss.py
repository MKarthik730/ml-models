from typing import List,Callable
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("capping_flooring_practice_multi.csv")
def loss_gradient(forward:dict[str,ndarray])->dict[str,ndarray]:
    dedy=-2*(forward['y']-forward['p'])
    deda = forward['x'].T @ dedy
    dedb=dedy.sum(axis=0)
    return {'a': deda, 'b': dedb}

loss_history=[]
iterations=[]

x=data[['price', 'rating']].values
y=data['quantity_sold'].values.reshape(-1,1)
features=x.shape[1]
x[:,0] = np.where(np.isnan(x[:,0]), np.nanmean(x[:,0]), x[:,0])
print()
a=np.ones((features,1))
b=np.array([2])
lr=0.001
for i in range(500):
    p=x@a+b
    forward={
    'x':x,
    'y':y,
    'p':p
    }
    loss = np.mean((y - p)**2)
    loss_history.append(loss)
    iterations.append(i)
    grad=loss_gradient(forward)
    a=a-lr*grad['a']
    b=b-lr*grad['b']
plt.plot(iterations,loss_history)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()
