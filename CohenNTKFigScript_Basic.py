# Submission - ready script for generating NTK overview / explanation figure from Cohen project

import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Module
from torch import nn
from torch.optim import SGD
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 9

torch.manual_seed(seed)

device = torch.device('cpu')

Din = 1
H = 100  # Hidden
Dout = 1

loss_fn = torch.nn.MSELoss()

# Training data
ndat = 6
ntest = 1000
xinnp = np.random.randn(ndat, Din)
xinnp[:,0] = np.linspace(-1,1,ndat)
xinnptest = np.zeros((ntest, Din))
xinnptest[:,0] = np.linspace(-1,1,ntest)
yinnp = np.zeros((ndat, Dout))
yinnptest = np.zeros((ntest, Dout))
# Generate actual target data
lr_use = 3e-2
for i in range(ndat):
    if xinnp[i, 0] > 0:
        yinnp[i,0] = 1
for i in range(ntest):
    if xinnptest[i, 0] > 0:
        yinnptest[i,0] = 1

device = torch.device("cpu")

xin = torch.from_numpy(xinnp)
xin = xin.to(torch.float).to(device)
yin = torch.from_numpy(yinnp)
yin = yin.to(torch.float).to(device)
xtest = torch.from_numpy(xinnptest)
xtest = xtest.to(torch.float).to(device)
ytest = torch.from_numpy(yinnptest)
ytest = ytest.to(torch.float).to(device)

class ShallowReLU(Module):
    def __init__(self):
        super(ShallowReLU, self).__init__()
        self.fc = nn.Linear(Din, H, bias=True)
        self.fc2 = nn.Linear(H, Dout, bias=True)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        y = self.fc(x)
        y = self.ReLU(y)
        y = self.fc2(y)
        return y

model = ShallowReLU()
model2 = ShallowReLU()

with torch.no_grad():
    model2.load_state_dict(model.state_dict())

numparams = 0
for j in model.parameters():
    numparams += j.flatten().size()[0]
print(numparams)

numiter = 1
numiter2 = 10000
printfreq = 100

optimizer = SGD(model.parameters(), lr=lr_use)

predict_yinit = model(xtest)
train_yinit = model(xin)
losses = np.zeros(numiter2)
testlosses = np.zeros(numiter2)


PNTK = torch.zeros((ndat,ntest)).to(device)
PNTK_raw = torch.zeros((ndat,ntest)).to(device)
NTKtrain = torch.zeros((ndat, numparams)).to(device)
NTKtest = torch.zeros((ntest, numparams)).to(device)

calculate_alphas = True
if calculate_alphas:
    dldysum = np.zeros((ndat,Dout))

startTime = time.time()
for t in range(numiter):

    # Get NTK components. Simple model, so done manually
    for i in range(ndat):
        optimizer.zero_grad()
        ytrain_i = model(xin[i])
        error_i = loss_fn(ytrain_i, yin[i])
        error_i.backward(retain_graph = True)
        pstart = 0
        for p in model.parameters():
            plen = len(p.flatten())
            NTKtrain[i,pstart:pstart+plen] = p.grad.data.flatten()
            pstart += plen
    for i in range(ntest):
        optimizer.zero_grad()
        ytest_i = model(xtest[i])
        ytest_i.backward(retain_graph = True)
        pstart = 0
        for p in model.parameters():
            plen = len(p.flatten())
            NTKtest[i,pstart:pstart+plen] = p.grad.data.flatten()
            pstart += plen

    optimizer.zero_grad()
    predict_y = model(xin)
    _error = loss_fn(predict_y, yin)
    losses[t] = _error.item()
    _error.backward(retain_graph = True)


    if calculate_alphas:
        dldy = torch.autograd.grad(outputs=_error, inputs=predict_y)[0]
        dldysum[:,:] += dldy.data.numpy()

    with torch.no_grad():
        dldysum2 = np.copy(dldysum)
        dldysum2[dldysum2==0] = 1
        NTKtrain_raw = (NTKtrain.clone()/dldysum2).to(torch.float32)
    PNTK     += -1 * lr_use * torch.einsum('ik, jk->ij', NTKtrain,     NTKtest) / ndat
    PNTK_raw += -1 * lr_use * torch.einsum('ik, jk->ij', NTKtrain_raw, NTKtest) / ndat

    optimizer.step()
    if t % printfreq == 0:
        print(t, _error.item())
    predict_ytest = model(xtest)
    test_error = loss_fn(predict_ytest, ytest)
    testlosses[t] = test_error.item()

endTime = time.time()
print('PNTK Train Time:' + str(endTime - startTime))

ymid = model(xtest).data.numpy().copy()

# Train until completion without NTK
for t in range(numiter, numiter2):

    optimizer.zero_grad()
    predict_y = model(xin)
    _error = loss_fn(predict_y, yin)
    losses[t] = _error.item()
    _error.backward(retain_graph = True)

    optimizer.step()
    if t % printfreq == 0:
        print(t, _error.item())
    predict_ytest = model(xtest)
    test_error = loss_fn(predict_ytest, ytest)
    testlosses[t] = test_error.item()


coloruse = np.zeros((ndat, 3))
coloruse[0,:] = [0,.8,.2]
coloruse[1,:] = [0,.6,.4]
coloruse[2,:] = [0,.4,.6]
coloruse[3,:] = [.2,.2,.8]
coloruse[4,:] = [.4,.2,.6]
coloruse[5,:] = [.6,.2,.4]

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize = (15,12))
for i in range(ndat):
    axs[0][1].plot(xinnptest, -1*PNTK_raw[i,:].data.numpy(), color = coloruse[i,:])
    axs[0][1].scatter(xin[i,0], yin[i,0], color = coloruse[i,:])
axs[0][1].set_title('NTK Influence')
axs[0][1].text(0, 0.75, r'K(x,$x_n$, $\theta_0$)', horizontalalignment = 'center', fontsize = 16)
axs[1][0].set_ylabel('Y')
for i in range(ndat):
    axs[1][0].plot(xinnptest, PNTK[i,:].data.numpy(), color = coloruse[i,:])
    axs[1][0].scatter(xin[i,0], yin[i,0], color = coloruse[i,:])
axs[1][0].set_title('NTK Changes')
axs[1][0].text(0, 0.65, r'$L^{\prime}$($y_n$,$\hat{y}_n(\theta_0)$)K(x,$x_n$, $\theta_0$)', horizontalalignment = 'center', fontsize = 16)

axs[0][0].plot(xinnptest, model2(xtest).data.numpy())
axs[0][0].plot(xinnptest, ymid)
axs[0][0].plot(xinnptest, model(xtest)[:,0].data.numpy())
axs[0][0].scatter(xin, yin, color = 'red')
axs[0][0].legend(['Initial Fit','1-step Fit','Final Fit','Train Data'])
axs[0][0].set_xlabel('X')
axs[0][0].set_ylabel('Y')
axs[0][0].set_title('Model Step')

axs[1][1].plot(xinnptest, model2(xtest)[:,0].data.numpy() + torch.sum(PNTK[:,:],0).data.numpy(), linewidth = 8, alpha = .5)
axs[1][1].plot(xinnptest, ymid)
axs[1][1].scatter(xin, yin, color = 'red')
axs[1][1].legend(['Initial Fit + NTK Changes','1-step Fit','Train Data'])
axs[1][1].set_xlabel('X')
axs[1][1].set_title('Model Step vs NTK Projection')
axs[1][1].text(0, 0.25, r'$\sum_{n \in \mathcal{D}_{train}}L^{\prime}$($y_n$,$\hat{y}_n(\theta_0)$)K(x,$x_n$, $\theta_0$) + $y_0$', horizontalalignment = 'center', fontsize = 16)
plt.suptitle('NTK Prediction', fontsize = 16)
plt.show()
