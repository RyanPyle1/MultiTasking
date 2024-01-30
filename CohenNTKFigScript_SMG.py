# Submission - ready script for generating SMG (task 1) figures from Cohen project

from copy import deepcopy
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Module
from torch import nn
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import pickle
import time
from sklearn import linear_model


def replicate_model(net_class, batch_size):
    class MultiNet(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.batch_size = batch_size
            self.model = net_class(**kwargs)

            if hasattr(self.model, 'input_size'):
                self.input_size = self.model.input_size

            models = []
            for _ in range(self.batch_size):
                models.append(deepcopy(self.model))
                for p, p_true in zip(models[-1].parameters(), self.model.parameters()):
                    p.data = p_true.data
            self.models = nn.ModuleList(models)

            self.parameters = self.model.parameters

        def forward(self, x):
            if self.training:
                assert x.size(0) == self.batch_size
                x = x.split(1, dim=0)
                y = torch.cat([model(x_i) for x_i, model in zip(x, self.models)], dim=0)
            else:
                y = self.model(x)

            return y

        def reduce_batch(self):
            """Puts per-example gradient in p.bgrad and aggregate on p.grad"""
            params = zip(*[model.parameters() for model in self.models])  # group per-layer
            for p, p_multi in zip(self.model.parameters(), params):
                p.bgrad = torch.stack([p_.grad for p_ in p_multi], dim=0)
                p.grad = torch.sum(p.bgrad, 0)
                for p_ in p_multi:
                    p_.grad = None

        def reassign_params(self):
            """Reassign parameters of sub-models to those of the main model"""
            for model in self.models:
                for p, p_true in zip(model.parameters(), self.model.parameters()):
                    p.data = p_true.data

        def get_detail(self, b):
            pass  # for compatibility with crb.nn.Module

    return MultiNet


seed = 4

torch.manual_seed(seed)

device = torch.device('cpu')

Din = 4
H = 3  # Hidden
Dout = 5

loss_fn = torch.nn.MSELoss()

import numpy as np


def whiten(X, fudge=1E-18):
    # the matrix X should be observations-by-components

    # get the covariance matrix
    Xcov = np.dot(X.T, X)

    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d + fudge))

    # whitening matrix
    W = np.dot(np.dot(V, D), V.T)

    # multiply by the whitening matrix
    X_white = np.dot(X, W)

    return X_white, W


# Generate data

Wtarget = np.random.rand(Din, Dout) * 10 - 20
constructW = True
if constructW:
    Wu, Ws, Wv = np.linalg.svd(Wtarget)
    Ws_construct = np.array([90, 24, 7, 1]) # good - get 2 groups done, with visible gap between
    Wtarget = np.zeros((Din, Dout))
    for i in range(4):
        Wtarget+= np.outer(Wu[:,i], Wv[i,:]) * Ws_construct[i]

Wu, Ws, Wv = np.linalg.svd(Wtarget)
W_approx = np.zeros((Din, Dout))
for i in range(3):
    W_approx+= np.outer(Wu[:,i], Wv[i,:]) * Ws[i]
    if i==0:
        W1 = np.copy(W_approx) # 1st mode
    if i==1:
        W2 = np.copy(W_approx) # 1st and second mode
    if i==2:
        W3 = np.copy(W_approx) # First 3 modes

ndat = 250
ntest = ndat
xinnp = np.random.randn(ndat, Din)
xinnp -= np.mean(xinnp,0)
xinnptest = np.copy(xinnp) # test data = train data
xinnp = whiten(xinnp)[0]
xinnptest = whiten(xinnptest)[0]

yinnp = (xinnp @ Wtarget)
yinnptest = (xinnptest @ Wtarget)
yinnp1 = (xinnp @ W1)
yinnp2 = (xinnp @ W2)
yinnp3 = (xinnp @ W3)
yinnptest1 = (xinnptest @ W1)
yinnptest2 = (xinnptest @ W2)
yinnptest3 = (xinnptest @ W3)

xin = torch.from_numpy(xinnp)
xin = xin.to(torch.float)
yin = torch.from_numpy(yinnp)
yin = yin.to(torch.float)
xtest = torch.from_numpy(xinnptest)
xtest = xtest.to(torch.float)
ytest = torch.from_numpy(yinnptest)
ytest = ytest.to(torch.float)

yin1 = torch.from_numpy(yinnp1)
yin1 = yin1.to(torch.float)
yin2 = torch.from_numpy(yinnp2)
yin2 = yin2.to(torch.float)
yin3 = torch.from_numpy(yinnp3)
yin3 = yin3.to(torch.float)
ytest1 = torch.from_numpy(yinnptest1)
ytest1 = ytest1.to(torch.float)
ytest2 = torch.from_numpy(yinnptest2)
ytest2 = ytest2.to(torch.float)
ytest3 = torch.from_numpy(yinnptest3)
ytest3 = ytest3.to(torch.float)


xycov = np.dot(yinnp.T, xinnp)
xycovtest = np.dot(yinnptest.T, xinnptest)
U,S,V = np.linalg.svd(xycov)

# Setup Model

class LinearRegression(Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(Din, H, bias=False)
        self.fc2 = nn.Linear(H, Dout, bias=False)

    def forward(self, x):
        y = self.fc(x)
        y = self.fc2(y)
        return y


model = LinearRegression()

numparams = 0
for j in model.parameters():
    numparams += j.flatten().size()[0]
print(numparams)

MNet = replicate_model(net_class=LinearRegression, batch_size=ndat)
model = MNet().to(device)
with torch.no_grad():
    model.model.fc.weight[:]*=.1
    model.model.fc2.weight[:]*=.1
Winit = model.model.fc.weight.data.numpy().T @ model.model.fc2.weight.data.numpy().T
FC1_init = np.copy(model.model.fc.weight.data.numpy())
FC2_init = np.copy(model.model.fc2.weight.data.numpy())

lr_use = 2e-1
numiter = 2500
printfreq = 50

optimizer = SGD(model.parameters(), lr=lr_use)

doNTK = True
doInfluence = True
storeNTKs = True
storefreq = 10

predict_yinit = model(xtest)
train_yinit = model(xin)

losses = np.zeros(numiter)
testlosses = np.zeros(numiter)
losses_comp = np.zeros((numiter, 3))
testlosses_comp = np.zeros((numiter, 3))

if doNTK:
    NTKtrain = np.zeros((Din, ndat, numparams, Dout))  # number of training points, number of network params, num outputs
    NTKtest = np.zeros((Din, ndat, numparams, Dout))

    if doInfluence:
        dldysum = np.zeros((ndat, Dout))
    if storeNTKs:
        NTKs = np.zeros((numiter//storefreq, Din, ndat, ntest, Dout))
        New_NTKs = np.zeros((numiter//storefreq, ndat, ntest, Dout, Dout))

startTime = time.time()
for t in range(numiter):
    if doNTK:  # Prep (per batch) NTKtest
        predict_y = model(xtest)
        yout2 = torch.sum(predict_y, 0)
        for q in range(Din):
            xtest_partial = torch.clone(xtest)
            for qq in range(Din):
                if qq!=q:
                    xtest_partial[:,qq]*=0
            predict_y = model(xtest_partial)  # batch_size * Dout
            yout2 = torch.sum(predict_y, 0)
            for j in range(Dout):
                optimizer.zero_grad()
                yout2[j].backward(retain_graph=True)
                model.reduce_batch()
                bgrads = [p.bgrad for p in model.parameters()]
                for i in range(ntest):
                    graduse = np.empty(0)
                    kuse = 0
                    for k in model.parameters():
                        graduse = np.append(graduse, bgrads[kuse][i].data.numpy().flatten())
                        kuse += 1
                    NTKtest[q, i, :, j] = graduse

    optimizer.zero_grad()
    predict_y = model(xin)
    _error = loss_fn(predict_y, yin)

    if doInfluence:
        loss_influences = torch.autograd.grad(outputs=_error, inputs=predict_y, retain_graph = True)[0]
        dldysum += loss_influences.squeeze().data.numpy()

    if doNTK:
        NTKtrain = np.zeros((Din, ndat, numparams, Dout))
        for q in range(Din):
            xin_partial = torch.clone(xin)
            for qq in range(Din):
                if qq!=q:
                    xin_partial[:,qq]*=0
            predict_y_partial = model(xin_partial)  # batch_size * Dout
            yout2 = torch.sum(predict_y_partial, 0)
            for j in range(Dout):
                optimizer.zero_grad()
                yout2[j].backward(retain_graph=True)
                model.reduce_batch()
                bgrads = [p.bgrad for p in model.parameters()]
                for i in range(ntest):
                    graduse = np.empty(0)
                    kuse = 0
                    for k in model.parameters():
                        graduse = np.append(graduse, bgrads[kuse][i].data.numpy().flatten())
                        kuse += 1
                    NTKtrain[q, i, :, j] = graduse

        train_3 = torch.einsum('ijkl,jl->ijkl', torch.tensor(NTKtrain).to(torch.float), loss_influences).data.numpy()
        NTKtrainv3 = np.sum(train_3,-1)
    losses[t] = _error.item()
    _error.backward()
    model.reduce_batch()


    # Store Previous Time Point's Model params
    W1last = torch.clone(model.model.fc.weight[:,:])
    W2last = torch.clone(model.model.fc2.weight[:,:])


    optimizer.step()
    model.reassign_params()



    if doNTK:
        NTK4 = -1 * lr_use * np.einsum('ik,qjkl->qijl', np.sum(NTKtrainv3, 0), NTKtest)
        NTK = np.copy(NTK4)
        if storeNTKs:
            if (t)%storefreq==0:
                NTKs[(t)//storefreq,:,:,:,:] = np.copy(NTK)
                New_NTK = -1*lr_use * np.einsum('ikl,jkm->ijlm', np.sum(NTKtest,0), np.sum(NTKtest,0))
                New_NTKs[(t)//storefreq,:,:,:,:] = np.copy(New_NTK)

    if t % printfreq == 0:
        print(t, _error.item())

    if doNTK:
        if t==0:
            FC1_1 = np.copy(model.model.fc.weight.data.numpy())
            FC2_1 = np.copy(model.model.fc2.weight.data.numpy())
            NTK_1 = np.copy(NTK)
            loss_influences_1 = np.copy(loss_influences.data.numpy())
            W_singleupdate = model.model.fc.weight.data.numpy().T @ model.model.fc2.weight.data.numpy().T

    predict_ytest = model.model(xtest)
    test_error = loss_fn(predict_ytest, ytest)
    testlosses[t] = test_error.item()

    predict_y = model.model(xin)
    train_error1 = loss_fn(predict_y, yin1)
    train_error2 = loss_fn(predict_y, yin2)
    train_error3 = loss_fn(predict_y, yin3)

    losses_comp[t, 0] = train_error1.item()
    losses_comp[t, 1] = train_error2.item()
    losses_comp[t, 2] = train_error3.item()

    predict_ytest = model.model(xtest)
    test_error1 = loss_fn(predict_ytest, ytest1)
    test_error2 = loss_fn(predict_ytest, ytest2)
    test_error3 = loss_fn(predict_ytest, ytest3)

    testlosses_comp[t, 0] = test_error1.item()
    testlosses_comp[t, 1] = test_error2.item()
    testlosses_comp[t, 2] = test_error3.item()

endTime = time.time()
print('Time:' + str(endTime - startTime))

plt.plot(losses)
plt.plot(testlosses, linestyle = '-.')
plt.legend(['Train', 'Test'], fontsize = 14)
plt.title('Mean Loss over Training', fontsize = 16)
plt.xlabel('Epoch', fontsize = 14)
plt.ylabel('Loss', fontsize = 14)
plt.show()


# First off, get actual learned Wout
Wout = model.model.fc.weight.data.numpy().T @ model.model.fc2.weight.data.numpy().T

sigma_approx = np.zeros((Dout,Din))
for i in range(3):
    sigma_approx += np.outer(U[:,i],V[i,:]) * S[i]

W_approx = np.zeros((Din, Dout))
Wu, Ws, Wv = np.linalg.svd(Wtarget)
for i in range(3):
    W_approx+= np.outer(Wu[:,i], Wv[i,:]) * Ws[i]
    print(np.mean(np.abs(W_approx - Wout)))
    print(np.mean(np.abs(xinnp @ W_approx - model.model(xin).data.numpy())))
    print('')

# Analyze our two chosen times
NTK1 = np.sum(np.sum(NTKs[19,:,:,:,:],1),1)
NTK2 = np.sum(np.sum(NTKs[77,:,:,:,:],1),1)
Wu, Ws, Wv = np.linalg.svd(Wtarget)
W_approx = np.zeros((Din, Dout))
for i in range(2):
    W_approx+= np.outer(Wu[:,i], Wv[i,:]) * Ws[i]
    if i==0:
        W1 = np.copy(W_approx) # 1st mode
    if i==1:
        W2 = np.copy(W_approx) # 1st and second mode
        W3 = W2 - W1 # second mode only
    if i==2:
        W3f = np.copy(W_approx)

W1u, W1s, W1v = np.linalg.svd(W1)
W2u, W2s, W2v = np.linalg.svd(W2)
W3u, W3s, W3v = np.linalg.svd(W3)

N1u, N1s, N1v = np.linalg.svd(NTK1)
N2u, N2s, N2v = np.linalg.svd(NTK2)

testsum = np.zeros((Din,Dout))
for i in range(Din):
    for k in range(Dout):
        for j in range(ndat):
            testsum[i,k]+= (NTK_1[i,j,:,k]/xinnp[:,i])[0] # very first NTK update, compare vs dW1


xin_dot = np.zeros((ndat, Din))
for i in range(ndat):
    for j in range(Din):
        xin_dot[i,j] = np.dot(xinnp[i,:],Wu[:,j])


test2U, test2S, test2V = np.linalg.svd(np.sum(np.sum(NTKs[19,:,:,:,:],0),-1))
test4U, test4S, test4V = np.linalg.svd(np.sum(np.sum(NTKs[77,:,:,:,:],0),-1))


# NTK t = early
corr = np.corrcoef(-1*test2U[:,0],np.abs(xin_dot[:,0]))[0,1]
fig, ax1 = plt.subplots()
ax1.set_xlabel('SV Vector Index / Input Index', fontsize = 14)
ax1.set_ylabel('Singular Vector Value', fontsize = 14, c = 'tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ln1 = ax1.plot(-1*test2U[:,0], label = 'eNTK Singular Vector')
ax2 = ax1.twinx()
ln2 = ax2.plot(np.abs(xin_dot[:,0]), c = 'tab:orange', label = 'Input $\cdot$ W Singular Vector')
ax2.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]])
ax2.set_ylabel('Input Projection', color='tab:orange', fontsize = 14)
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.suptitle('Input Mode 1 vs eNTK(190) Mode 1, $\Sigma=$ {0:.3g}'.format(corr), fontsize = 14)
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize = 14)
plt.show()

# NTK t = late
corr = np.corrcoef(np.abs(test4U[:,1]),np.abs(xin_dot[:,1]))[0,1]
fig, ax1 = plt.subplots()
ax1.set_xlabel('SV Vector Index / Input Index', fontsize = 14)
ax1.set_ylabel('Singular Vector Value', fontsize = 14, c = 'tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ln1 = ax1.plot(np.abs(test4U[:,1]), label = 'eNTK Singular Vector')
ax2 = ax1.twinx()
ln2 = ax2.plot(np.abs(xin_dot[:,1]), c = 'tab:orange', label = 'Input $\cdot$ W Singular Vector')
ax2.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]])
ax2.set_ylabel('Input Projection', color='tab:orange', fontsize = 14)
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.suptitle('Input Mode 2 vs eNTK(770) Mode 2, $\Sigma=$ {0:.3g}'.format(corr), fontsize = 14)
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize = 14)
plt.show()


dotprods = np.zeros((numiter//10,20))
for t in range(numiter//10):
    tuse = t

    testsumN = np.zeros((Din, Dout))
    for i in range(Din):
        for k in range(Dout):
            for j in range(ndat):
                testsumN[i, k] += (NTKs[tuse, i, j, :, k] / xinnp[:, i])[0]

    TSNu, TSNs, TSNv = np.linalg.svd(testsumN)
    # 5-8 : NTK primary learning
    dotprods[t, 4] = np.dot(TSNu[:,0], Wu[:,0])
    dotprods[t, 5] = np.dot(TSNu[:,0], Wu[:,1])
    dotprods[t, 6] = np.dot(TSNu[:,0], Wu[:,2])
    dotprods[t, 7] = np.dot(TSNu[:,0], Wu[:,3])
    # 9-12 : NTK secondary learning
    dotprods[t, 8] = np.dot(TSNu[:, 1], Wu[:, 0])
    dotprods[t, 9] = np.dot(TSNu[:, 1], Wu[:, 1])
    dotprods[t, 10] = np.dot(TSNu[:, 1], Wu[:, 2])
    dotprods[t, 11] = np.dot(TSNu[:, 1], Wu[:, 3])
    # 13-16 : NTK tertiary learning
    dotprods[t, 12] = np.dot(TSNu[:, 2], Wu[:, 0])
    dotprods[t, 13] = np.dot(TSNu[:, 2], Wu[:, 1])
    dotprods[t, 14] = np.dot(TSNu[:, 2], Wu[:, 2])
    dotprods[t, 15] = np.dot(TSNu[:, 2], Wu[:, 3])


# New : try to combine loss, PNTK learning prediction plots to show overlaps...

plt.plot(np.arange(0,2500,10), np.abs(dotprods[:,4:8]))
plt.legend(['SV 1 Learning', 'SV 2 Learning', 'SV 3 Learning','SV 4 Learning'], fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.ylabel('Dot Product', fontsize = 14)
plt.title('Primary eNTK Learning', fontsize = 16)
plt.show()

plt.plot(np.arange(0,2500,10), np.abs(dotprods[:,8:12]))
plt.legend(['SV 1 Learning', 'SV 2 Learning', 'SV 3 Learning','SV 4 Learning'], fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.ylabel('Dot Product', fontsize = 14)
plt.title('Secondary eNTK Learning', fontsize = 16)
plt.show()

# Two main plots there - one of Svec matching, one of Svals (or something related to Svals..?)
# Og code for SVec one:
fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch', fontsize = 14)
ax1.set_ylabel('Loss', color='red', fontsize = 14)
ax1.semilogy(np.arange(numiter), losses, color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax2 = ax1.twinx()
ax2.set_ylabel('Primary eNTK Learning', color='blue', fontsize = 14)
ax2.plot(np.arange(1,numiter,10), np.abs(dotprods[:,4]), color=[0,  0, 1, .8])
ax2.plot(np.arange(1,numiter,10), np.abs(dotprods[:,5]), color=[.1,.4,.8, .8])
ax2.plot(np.arange(1,numiter,10), np.abs(dotprods[:,6]), color=[.2,.8,.6, .8])
ax2.legend(['SVec 1','SVec 2','SVec 3'], fontsize = 14)
ax2.tick_params(axis='y', labelcolor='blue')
ax2.vlines([190,770], -1, 2, colors = 'k', alpha = .5, linestyle = 'dotted')
ax2.set_ylim([-.03,1.03])
ax2.text(190-100,1.05,'t=190')
ax2.text(770-100,1.05,'t=770')
plt.suptitle('Training Loss and Singular Vector Learning over Training', fontsize = 16)
plt.show()

new_dotprods = np.zeros((250,6))
new_svs = np.zeros((250,6))
toSVD = np.zeros((250,250))
toSVD2 = np.zeros((250,250))
for i in range(250):
    toSVD = np.sum(np.sum(New_NTKs[i,:,:,:,:],-1),-1)
    toSVD2 += np.sum(np.sum(New_NTKs[i,:,:,:,:],-1),-1)
    newTU, newTS, newTV = np.linalg.svd(toSVD)
    newTU2, newTS2, newTV2 = np.linalg.svd(toSVD2)
    A = newTU[:,0]
    B = xin_dot[:,0]
    new_dotprods[i,0] = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    B = xin_dot[:, 1]
    new_dotprods[i, 1] = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    B = xin_dot[:, 2]
    new_dotprods[i, 2] = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    A = newTU[:, 0]
    B = xin_dot[:, 0]
    new_dotprods[i, 3] = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    A = newTU[:, 1]
    B = xin_dot[:, 1]
    new_dotprods[i, 4] = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    A = newTU[:, 2]
    B = xin_dot[:, 2]
    new_dotprods[i, 5] = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    new_svs[i, 0] = newTS[0]
    new_svs[i, 1] = newTS[1]
    new_svs[i, 2] = newTS[2]
    new_svs[i, 3] = newTS2[0]
    new_svs[i, 4] = newTS2[1]
    new_svs[i, 5] = newTS2[2]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch', fontsize = 14)
ax1.set_ylabel('Loss', color='red', fontsize = 14)
ax1.semilogy(np.arange(numiter), losses, color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax2 = ax1.twinx()
ax2.set_ylabel('Primary NTK Singular Vectors', color='blue', fontsize = 14)
ax2.plot(np.arange(1,numiter,10), np.abs(new_dotprods[:,0]), color=[0,  0, 1, .8])
ax2.plot(np.arange(1,numiter,10), np.abs(new_dotprods[:,1]), color=[.1,.4,.8, .8])
ax2.plot(np.arange(1,numiter,10), np.abs(new_dotprods[:,2]), color=[.2,.8,.6, .8])
ax2.legend(['SVec 1','SVec 2','SVec 3'], fontsize = 14)
ax2.tick_params(axis='y', labelcolor='blue')
ax2.vlines([190,770], -1, 2, colors = 'k', alpha = .5, linestyle = 'dotted')
ax2.set_ylim([-.03,1.03])
ax2.text(190-100,1.05,'t=190')
ax2.text(770-100,1.05,'t=770')
plt.suptitle('Training Loss and Singular Vector Activity over Training', fontsize = 16)
plt.show()

fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch', fontsize = 14)
ax1.set_ylabel('Loss', color='red', fontsize = 14)
#ax1.plot(np.arange(numiter), losses, color='red')
ax1.semilogy(np.arange(numiter), losses, color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax2 = ax1.twinx()
ax2.set_ylabel('NTK Singular Values', color='blue', fontsize = 14)
ax2.plot(np.arange(1,numiter,10), new_svs[:,0], color=[0,  0, 1, .8])
ax2.plot(np.arange(1,numiter,10), new_svs[:,1], color=[.1,.4,.8, .8])
ax2.plot(np.arange(1,numiter,10), new_svs[:,2], color=[.2,.8,.6, .8])
ax2.legend(['SVal 1','SVal 2','SVal 3'], fontsize = 14)
ax2.tick_params(axis='y', labelcolor='blue')
ax2.vlines([190,770], -np.max(new_svs[:,:2]), 2*np.max(new_svs[:,:2]), colors = 'k', alpha = .5, linestyle = 'dotted')
ax2.set_ylim([np.max(new_svs[:,:2])*-.03, 1.03*np.max(new_svs[:,:2])])
ax2.text(190-100,1.05*np.max(new_svs[:,:2]),'t=190')
ax2.text(770-100,1.05*np.max(new_svs[:,:2]),'t=770')
plt.suptitle('Training Loss and NTK Singular Values over Training', fontsize = 16)
plt.show()