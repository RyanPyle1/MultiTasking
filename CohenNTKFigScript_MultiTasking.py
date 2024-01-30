# Submission - ready script for generating Multitasking (task 2) figures from Cohen project

# Script is intended to be run multiple times.
# A single run generates either training or fine-tuning data for both the std and large init
# Fine-tuning requires previously running training, and then loading that data
# Option: doMPHATE = False by default. Does MPHATE analysis if true, but this can be quite slow
# Option: lr_adjust = True OR do_long = True
# do_long does a longer run with no LR tuning (abbreviated as ET - extended time, when saving data)
# lr_adjust does a shorter run that mimics the longer run by scaling LR times unevenly between the std/large init cases

import torch.utils.data
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import torch.nn as nn
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

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
                y = torch.cat([model(x_i)[0] for x_i, model in zip(x, self.models)], dim=0)
                y2 = torch.cat([model(x_i)[1] for x_i, model in zip(x, self.models)], dim=0)
                return y, y2
            else:
                y = self.model(x)
                return y

            #return y

        def reduce_batch(self):
            """Puts per-example gradient in p.bgrad and aggregate on p.grad"""
            params = zip(*[model.parameters() for model in self.models])  # group per-layer
            for p, p_multi in zip(self.model.parameters(), params):
                p.bgrad = torch.stack([p_.grad for p_ in p_multi], dim=0)
                p.grad = torch.sum(p.bgrad,0)
                for p_ in p_multi:
                    p_.grad = None

        def reassign_params(self):
            """Reassign parameters of sub-models to those of the main model"""
            for model in self.models:
                for p, p_true in zip(model.parameters(), self.model.parameters()):
                    p.data = p_true.data

        def get_detail(self, b): pass  # for compatibility with crb.nn.Module

    return MultiNet

# Std settings
Ningroup = 4
Noutgroup = 3
Nfeatures = 3

Din = (Ningroup*Nfeatures) + (Ningroup * Noutgroup)
Dout = Noutgroup * Nfeatures
N = 500 # smallish network for NTK analysis
Ntest = N
H = 200 # hidden size


device = 'cpu'


# For base train
Single12All = True
Multi12All = False
# For finetune
#Single12All = False
#Multi12All = True


# Generate training data:
x = torch.zeros((N + Ntest, Din))
y = torch.zeros((N + Ntest, Dout))

from sympy.utilities.iterables import multiset_permutations
base = np.zeros(Nfeatures)
base[0] = 1
xfull = np.zeros((Nfeatures+1, Nfeatures))
xfull[-1,:] = np.zeros(Nfeatures)
iter = 0
for p in multiset_permutations(base):
    xfull[iter,:] = p
    iter+=1
for i in range(N + Ntest):
    if i == N:  # Can change task generation settings for testing data here
        p = .25  # Task probability
        # Train
        Single12All = True
        Multi12All = False
        # Finetune
        #Single12All = False
        #Multi12All = True
    indices = np.random.choice(Nfeatures, Ningroup, replace=True)
    xinp = xfull[indices, :].flatten()  # Ningroup sets of random length-3 one hot vectors
    if Single12All:
        xtask = np.random.permutation([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        taskids = np.zeros((12, 12))
        for j in range(12):
            taskids[j, j] = 1
    if Multi12All: # randomly choose 1,2, or 3 tasks simultaneously
        randnum = np.random.rand()
        if randnum < .33:
            xtask = np.random.permutation([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif randnum < .66:
            indices = np.random.choice(3, 2, replace=False)
            indices = np.insert(indices, np.random.randint(0, 3), 3)
            indices = np.insert(indices, np.random.randint(0, 4), 3)
            xtask = xfull[indices,
                    :].flatten()  # 2 sets of random length-3 one hot vectors, non-repeating, with one set of all 0s
        else:
            indices = np.random.choice(3, 3, replace=False)
            indices = np.insert(indices, np.random.randint(0, 4), 3)
            xtask = xfull[indices, :].flatten()  # 3 sets of random length-3 one hot vectors, non-repeating
        # Generate taskids
        taskids = np.zeros((12 + 36 + 24, 12))
        for j in range(12):  # 12 single tasks
            taskids[j, j] = 1
        for j in range(36):  # how many double tasks? 6*6 = 36
            # 6 possibilites for each 1-2-3 set: 12,13,21,23,31,32
            # 6 possibilities for the 00: 00xx, 0x0x, 0xx0, x00x, x0x0, xx00
            if j < 6:
                taskids[12 + j, 0:6] = np.zeros(6)
                taskids[12 + j, 6:9] = xfull[j // 2, :]  # 0 0 1 1 2 2
                taskids[12 + j, 9:12] = xfull[((j // 2) + 1 + (j % 2)) % 3, :]  # 1 2 2 0 0 1
            elif j < 12:
                taskids[12 + j, 0:3] = np.zeros(3)
                taskids[12 + j, 6:9] = np.zeros(3)
                taskids[12 + j, 3:6] = xfull[(j - 6) // 2, :]  # 0 0 1 1 2 2
                taskids[12 + j, 9:12] = xfull[(((j - 6) // 2) + 1 + (j % 2)) % 3, :]  # 1 2 2 0 0 1
            elif j < 18:
                taskids[12 + j, 0:3] = np.zeros(3)
                taskids[12 + j, 9:12] = np.zeros(3)
                taskids[12 + j, 3:6] = xfull[(j - 12) // 2, :]  # 0 0 1 1 2 2
                taskids[12 + j, 6:9] = xfull[(((j - 12) // 2) + 1 + (j % 2)) % 3, :]  # 1 2 2 0 0 1
            elif j < 24:
                taskids[12 + j, 3:9] = np.zeros(6)
                taskids[12 + j, 0:3] = xfull[(j - 18) // 2, :]  # 0 0 1 1 2 2
                taskids[12 + j, 9:12] = xfull[(((j - 18) // 2) + 1 + (j % 2)) % 3, :]  # 1 2 2 0 0 1
            elif j < 30:
                taskids[12 + j, 3:6] = np.zeros(3)
                taskids[12 + j, 9:12] = np.zeros(3)
                taskids[12 + j, 0:3] = xfull[(j - 24) // 2, :]  # 0 0 1 1 2 2
                taskids[12 + j, 6:9] = xfull[(((j - 24) // 2) + 1 + (j % 2)) % 3, :]  # 1 2 2 0 0 1
            else:
                taskids[12 + j, 6:12] = np.zeros(6)
                taskids[12 + j, 0:3] = xfull[(j - 30) // 2, :]  # 0 0 1 1 2 2
                taskids[12 + j, 3:6] = xfull[(((j - 30) // 2) + 1 + (j % 2)) % 3, :]  # 1 2 2 0 0 1
        from sympy.utilities.iterables import multiset_permutations

        # for j in range(16): # Triples: 0,1,2,3
        # 0123, 0132, 0213, 0231, 0312, 0321 * 4 =24
        j = 0
        for q in multiset_permutations([0, 1, 2, 3]):
            taskids[12 + 36 + j, 0:3] = xfull[q[0]]
            taskids[12 + 36 + j, 3:6] = xfull[q[1]]
            taskids[12 + 36 + j, 6:9] = xfull[q[2]]
            taskids[12 + 36 + j, 9:12] = xfull[q[3]]
            j += 1
    # Generate total input
    x[i, :(Ningroup*Nfeatures)] = torch.from_numpy(xinp)
    x[i, (Ningroup*Nfeatures):] = torch.from_numpy(xtask)
    # Generate outputs
    yout = np.zeros(Noutgroup*Nfeatures)
    if Ningroup == 6: # assumes 6/4 setup
        yout[:3] = xtask[0] * xinp[:3] + xtask[4] * xinp[3:6] + xtask[8] * xinp[6:9] + xtask[12] * xinp[9:12] + xtask[16] * xinp[12:15] + xtask[20] * xinp[15:18]
        yout[3:6] = xtask[1] * xinp[:3] + xtask[5] * xinp[3:6] + xtask[9] * xinp[6:9] + xtask[13] * xinp[9:12] + xtask[17] * xinp[12:15] + xtask[21] * xinp[15:18]
        yout[6:9] = xtask[2] * xinp[:3] + xtask[6] * xinp[3:6] + xtask[10] * xinp[6:9] + xtask[14] * xinp[9:12] + xtask[18] * xinp[12:15] + xtask[22] * xinp[15:18]
        yout[9:12] = xtask[3] * xinp[:3] + xtask[7] * xinp[3:6] + xtask[11] * xinp[6:9] + xtask[15] * xinp[9:12] + xtask[19] * xinp[12:15] + xtask[23] * xinp[15:18]
    if Ningroup == 4: # 4/3 setup
        yout[:Nfeatures]            = xtask[0] * xinp[:Nfeatures] + xtask[3] * xinp[Nfeatures:2*Nfeatures] + xtask[6] * xinp[2*Nfeatures:3*Nfeatures] + xtask[9] * xinp[3*Nfeatures:]
        yout[Nfeatures:2*Nfeatures] = xtask[1] * xinp[:Nfeatures] + xtask[4] * xinp[Nfeatures:2*Nfeatures] + xtask[7] * xinp[2*Nfeatures:3*Nfeatures] + xtask[10] * xinp[3*Nfeatures:]
        yout[2*Nfeatures:]          = xtask[2] * xinp[:Nfeatures] + xtask[5] * xinp[Nfeatures:2*Nfeatures] + xtask[8] * xinp[2*Nfeatures:3*Nfeatures] + xtask[11] * xinp[3*Nfeatures:]
    y[i, :] = torch.from_numpy(yout)

xtrain = x[:N, :]
ytrain = y[:N, :]
xtest = x[-Ntest:, :]
ytest = y[-Ntest:, :]

# Get task index using np.where((taskids == xtask).all(axis=1))[0][0]
# In the inputs, xtask = x[:,9:]
# e.g. (np.where((taskids == x[i,9:].numpy()).all(axis=1))[0][0])
tstore = np.zeros(N + Ntest)  # indicator of which task is used
xactstore = np.zeros(N + Ntest) # indicator of which input is the ACTIVE input (e.g. the one to be routed) - only meaningful for singletask
outstore = np.zeros(N + Ntest) # indicator of which output is the target to be routed to - only meaningful for singletask
for i in range(N + Ntest):
    tstore[i] = np.where((taskids == x[i, (Ningroup*Nfeatures):].numpy()).all(axis=1))[0][0]
    if Ningroup==6:
        if tstore[i]<24:
            outstore[i] = np.argmax(y[i,:]).item()
        if tstore[i]<4:
            xactstore[i] = np.where(x[i,:3]!= 0)[0][0]
        elif tstore[i]<8:
            xactstore[i] = np.where(x[i, 3:6] != 0)[0][0] + 3
        elif tstore[i]<12:
            xactstore[i] = np.where(x[i, 6:9] != 0)[0][0] + 6
        elif tstore[i]<16:
            xactstore[i] = np.where(x[i, 9:12] != 0)[0][0] + 9
        elif tstore[i]<20:
            xactstore[i] = np.where(x[i, 12:15] != 0)[0][0] + 12
        elif tstore[i]<24:
            xactstore[i] = np.where(x[i, 15:18] != 0)[0][0] + 15
    if Ningroup==4:
        if tstore[i]<12:
            outstore[i] = np.argmax(y[i,:]).item()
        if tstore[i]<3:
            xactstore[i] = np.where(x[i,:Nfeatures]!= 0)[0][0]
        elif tstore[i]<6:
            xactstore[i] = np.where(x[i, Nfeatures:2*Nfeatures] != 0)[0][0] + Nfeatures
        elif tstore[i]<9:
            xactstore[i] = np.where(x[i, 2*Nfeatures:3*Nfeatures] != 0)[0][0] + 2*Nfeatures
        elif tstore[i]<12:
            xactstore[i] = np.where(x[i, 3*Nfeatures:4*Nfeatures] != 0)[0][0] + 3*Nfeatures
        elif tstore[i] < (12 + 36):  # Two tasks
            xactstore[i] = (Ningroup*Nfeatures)
            outstore[i] = (Noutgroup*Nfeatures)
        elif tstore[i] < (12 + 36 + 24):  # Three tasks
            xactstore[i] = (Ningroup*Nfeatures)+1
            outstore[i] = (Noutgroup*Nfeatures)+1


from torch import nn

class CohenNet(nn.Module):
    def __init__(self):
        super(CohenNet, self).__init__()
        self.fc1 = nn.Linear(Din, H)  # set up first standard FC layer
        self.fc2 = nn.Linear(H + Ningroup*Noutgroup, Dout)  # set up the other standard  FC layer

    def forward(self, input):
        f1 = self.fc1(input)
        f1 = torch.sigmoid(f1)
        f1combo = torch.cat((f1, input[:, (Ningroup*Nfeatures):]), 1)  # For batch learning
        f2 = self.fc2(f1combo)
        f2 = torch.sigmoid(f2)
        return f2, f1  # output, hidden

# Cohen Init
MNet = replicate_model(net_class=CohenNet, batch_size=N)
model = MNet().to(device)
MNet2 = replicate_model(net_class = CohenNet, batch_size = N)
model2 = MNet().to(device)
model2.load_state_dict(model.state_dict())
with torch.no_grad():
    weight_scale = .1  # uniform from -w_s to + w_s
    model.model.fc1.weight[:, :] = torch.rand((H, Din)) * (2 * weight_scale) - weight_scale
    model.model.fc2.weight[:, :] = torch.rand((Dout, H + Ningroup*Noutgroup)) * (2 * weight_scale) - weight_scale
    model2.model.fc1.weight[:,:] = model.model.fc1.weight[:,:].clone() * 10
    model2.model.fc2.weight[:, :] = model.model.fc2.weight[:, :].clone() * 10

    bias_init = -2  # og: -2, with sigmoid activation
    model.model.fc1.bias[:] = torch.ones(H) * bias_init
    model.model.fc2.bias[:] = torch.ones(Dout) * bias_init
    model2.model.fc1.bias[:] = torch.ones(H) * bias_init
    model2.model.fc2.bias[:] = torch.ones(Dout) * bias_init
model.reassign_params()
model2.reassign_params()

numparams = 0
for j in model.model.parameters():
    if j.requires_grad == True:
        if j.dim()==2: # only count the weights, not biases
            numparams += j.flatten().size()[0]

loss_fn = torch.nn.MSELoss(size_average=False) # e.g. just summed

# Load models, if desired
# Note here, std / large refers not to model but model pair
# std is std runtime, tuned LRs
# large is larger runtime, untuned LRs
# model is std init, model2 is large init
loadmodels_LRRep1 = False
loadmodels_ETRep1 = False
loadmodels_LRRep2 = False
loadmodels_ETRep2 = False
loadmodels_LRRep3 = False
loadmodels_ETRep3 = False
# Model LR Reps 4-10
loadmodels_LRRep4 = False
loadmodels_LRRep5 = False
loadmodels_LRRep6 = False
loadmodels_LRRep7 = False
loadmodels_LRRep8 = False
loadmodels_LRRep9 = False
loadmodels_LRRep10 = False
# Model ET Reps 4-10
loadmodels_ETRep4 = False
loadmodels_ETRep5 = False
loadmodels_ETRep6 = False
loadmodels_ETRep7 = False
loadmodels_ETRep8 = False
loadmodels_ETRep9 = False
loadmodels_ETRep10 = False

if loadmodels_LRRep1:
    StdAllTrain11LRRep1 = f'data\Pickle\CohenStdAllTrain11LRRep1.pt'
    LargeAllTrain11LRRep1 = f'data\Pickle\CohenLargeAllTrain11LRRep1.pt'
    model.load_state_dict(torch.load(StdAllTrain11LRRep1))
    model2.load_state_dict(torch.load(LargeAllTrain11LRRep1))
if loadmodels_ETRep1:
    StdAllTrain11ETRep1 = f'data\Pickle\CohenStdAllTrain11ETRep1.pt'
    LargeAllTrain11ETRep1 = f'data\Pickle\CohenLargeAllTrain11ETRep1.pt'
    model.load_state_dict(torch.load(StdAllTrain11ETRep1))
    model2.load_state_dict(torch.load(LargeAllTrain11ETRep1))
if loadmodels_LRRep2:
    StdAllTrain11LRRep2 = f'data\Pickle\CohenStdAllTrain11LRRep2.pt'
    LargeAllTrain11LRRep2 = f'data\Pickle\CohenLargeAllTrain11LRRep2.pt'
    model.load_state_dict(torch.load(StdAllTrain11LRRep2))
    model2.load_state_dict(torch.load(LargeAllTrain11LRRep2))
if loadmodels_ETRep2:
    StdAllTrain11ETRep2 = f'data\Pickle\CohenStdAllTrain11ETRep2.pt'
    LargeAllTrain11ETRep2 = f'data\Pickle\CohenLargeAllTrain11ETRep2.pt'
    model.load_state_dict(torch.load(StdAllTrain11ETRep2))
    model2.load_state_dict(torch.load(LargeAllTrain11ETRep2))
if loadmodels_LRRep3:
    StdAllTrain11LRRep3 = f'data\Pickle\CohenStdAllTrain11LRRep3.pt'
    LargeAllTrain11LRRep3 = f'data\Pickle\CohenLargeAllTrain11LRRep3.pt'
    model.load_state_dict(torch.load(StdAllTrain11LRRep3))
    model2.load_state_dict(torch.load(LargeAllTrain11LRRep3))
if loadmodels_ETRep3:
    StdAllTrain11ETRep3 = f'data\Pickle\CohenStdAllTrain11ETRep3.pt'
    LargeAllTrain11ETRep3 = f'data\Pickle\CohenLargeAllTrain11ETRep3.pt'
    model.load_state_dict(torch.load(StdAllTrain11ETRep3))
    model2.load_state_dict(torch.load(LargeAllTrain11ETRep3))

if loadmodels_LRRep4:
    StdAllTrain11LRRep14 = f'data\Pickle\CohenStdAllTrain11LRRep14.pt'
    LargeAllTrain11LRRep14 = f'data\Pickle\CohenLargeAllTrain11LRRep14.pt'
    model.load_state_dict(torch.load(StdAllTrain11LRRep14))
    model2.load_state_dict(torch.load(LargeAllTrain11LRRep14))
if loadmodels_LRRep5:
    StdAllTrain11LRRep15 = f'data\Pickle\CohenStdAllTrain11LRRep15.pt'
    LargeAllTrain11LRRep15 = f'data\Pickle\CohenLargeAllTrain11LRRep15.pt'
    model.load_state_dict(torch.load(StdAllTrain11LRRep15))
    model2.load_state_dict(torch.load(LargeAllTrain11LRRep15))
if loadmodels_LRRep6:
    StdAllTrain11LRRep16 = f'data\Pickle\CohenStdAllTrain11LRRep16.pt'
    LargeAllTrain11LRRep16 = f'data\Pickle\CohenLargeAllTrain11LRRep16.pt'
    model.load_state_dict(torch.load(StdAllTrain11LRRep16))
    model2.load_state_dict(torch.load(LargeAllTrain11LRRep16))
if loadmodels_LRRep7:
    StdAllTrain11LRRep17 = f'data\Pickle\CohenStdAllTrain11LRRep17.pt'
    LargeAllTrain11LRRep17 = f'data\Pickle\CohenLargeAllTrain11LRRep17.pt'
    model.load_state_dict(torch.load(StdAllTrain11LRRep17))
    model2.load_state_dict(torch.load(LargeAllTrain11LRRep17))
if loadmodels_LRRep8:
    StdAllTrain11LRRep18 = f'data\Pickle\CohenStdAllTrain11LRRep18.pt'
    LargeAllTrain11LRRep18 = f'data\Pickle\CohenLargeAllTrain11LRRep18.pt'
    model.load_state_dict(torch.load(StdAllTrain11LRRep18))
    model2.load_state_dict(torch.load(LargeAllTrain11LRRep18))
if loadmodels_LRRep9:
    StdAllTrain11LRRep19 = f'data\Pickle\CohenStdAllTrain11LRRep19.pt'
    LargeAllTrain11LRRep19 = f'data\Pickle\CohenLargeAllTrain11LRRep19.pt'
    model.load_state_dict(torch.load(StdAllTrain11LRRep19))
    model2.load_state_dict(torch.load(LargeAllTrain11LRRep19))
if loadmodels_LRRep10:
    StdAllTrain11LRRep20 = f'data\Pickle\CohenStdAllTrain11LRRep20.pt'
    LargeAllTrain11LRRep20 = f'data\Pickle\CohenLargeAllTrain11LRRep20.pt'
    model.load_state_dict(torch.load(StdAllTrain11LRRep20))
    model2.load_state_dict(torch.load(LargeAllTrain11LRRep20))


if loadmodels_ETRep4:
    StdAllTrain11ETRep14 = f'data\Pickle\CohenStdAllTrain11ETRep14.pt'
    LargeAllTrain11ETRep14 = f'data\Pickle\CohenLargeAllTrain11ETRep14.pt'
    model.load_state_dict(torch.load(StdAllTrain11ETRep14))
    model2.load_state_dict(torch.load(LargeAllTrain11ETRep14))
if loadmodels_ETRep5:
    StdAllTrain11ETRep15 = f'data\Pickle\CohenStdAllTrain11ETRep15.pt'
    LargeAllTrain11ETRep15 = f'data\Pickle\CohenLargeAllTrain11ETRep15.pt'
    model.load_state_dict(torch.load(StdAllTrain11ETRep15))
    model2.load_state_dict(torch.load(LargeAllTrain11ETRep15))
if loadmodels_ETRep6:
    StdAllTrain11ETRep16 = f'data\Pickle\CohenStdAllTrain11ETRep16.pt'
    LargeAllTrain11ETRep16 = f'data\Pickle\CohenLargeAllTrain11ETRep16.pt'
    model.load_state_dict(torch.load(StdAllTrain11ETRep16))
    model2.load_state_dict(torch.load(LargeAllTrain11ETRep16))
if loadmodels_ETRep7:
    StdAllTrain11ETRep17 = f'data\Pickle\CohenStdAllTrain11ETRep17.pt'
    LargeAllTrain11ETRep17 = f'data\Pickle\CohenLargeAllTrain11ETRep17.pt'
    model.load_state_dict(torch.load(StdAllTrain11ETRep17))
    model2.load_state_dict(torch.load(LargeAllTrain11ETRep17))
if loadmodels_ETRep8:
    StdAllTrain11ETRep18 = f'data\Pickle\CohenStdAllTrain11ETRep18.pt'
    LargeAllTrain11ETRep18 = f'data\Pickle\CohenLargeAllTrain11ETRep18.pt'
    model.load_state_dict(torch.load(StdAllTrain11ETRep18))
    model2.load_state_dict(torch.load(LargeAllTrain11ETRep18))
if loadmodels_ETRep9:
    StdAllTrain11ETRep19 = f'data\Pickle\CohenStdAllTrain11ETRep19.pt'
    LargeAllTrain11ETRep19 = f'data\Pickle\CohenLargeAllTrain11ETRep19.pt'
    model.load_state_dict(torch.load(StdAllTrain11ETRep19))
    model2.load_state_dict(torch.load(LargeAllTrain11ETRep19))
if loadmodels_ETRep10:
    StdAllTrain11ETRep20 = f'data\Pickle\CohenStdAllTrain11ETRep20.pt'
    LargeAllTrain11ETRep20 = f'data\Pickle\CohenLargeAllTrain11ETRep20.pt'
    model.load_state_dict(torch.load(StdAllTrain11ETRep20))
    model2.load_state_dict(torch.load(LargeAllTrain11ETRep20))



learning_rate = 2e-4
learning_rate2 = 2e-4
PrintFreq = 10


# Try accelerated LR for std init single/single, and finetuning
learning_rate = .01
learning_rate2 = .01
lr_adjust = True
if lr_adjust:
    learning_rate*=3 # *10 is a bit too large, *3 works
    learning_rate2*=1 #*10 is far too large, *3 is a bit too large
    PlotFreq = 25
    NumIters = 2500

do_long = False
if do_long:
    PlotFreq = 100
    NumIters = 10000

numplot = NumIters//PlotFreq + 1
hstore = np.zeros((numplot,N,H))
lstore = np.zeros(numplot)
hstore2 = np.zeros((numplot,N,H))
lstore2 = np.zeros(numplot)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
model.float()

optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate2)
model2.float()

# NTK setup
numeigvec = 10
NTK = np.zeros((Ntest,N,Dout))
NTK2 = np.zeros((Ntest,N,Dout))

taskinds = np.argsort(tstore)  # ascending, from task 0 to 8
taskinds_train = np.argsort(tstore[:500])
taskinds_test = np.argsort(tstore[500:])
datainds = np.argsort(xactstore)
datainds_train = np.argsort(xactstore[:500])
datainds_test = np.argsort(xactstore[500:])
outinds = np.argsort(outstore)
outinds_train = np.argsort(outstore[:500])
outinds_test = np.argsort(outstore[500:])
ntask = taskids.shape[0]
tsum = np.zeros(ntask)
for i in range(ntask):
    tsum[i] = np.sum(tstore[:500] == i)
dsum = np.zeros(len(np.unique(xactstore)))
for i in range(len(np.unique(xactstore))):
    dsum[i] = np.sum(xactstore[:500] == i)
nout = len(np.unique(outstore))  # Dout or Dout+2 if multi
osum = np.zeros(nout)
for i in range(nout):
    osum[i] = np.sum(outstore[:500] == i)

Dinuse = Din
Huse = H + Ningroup*3

taskunitinds = []
for i in range(12):
    for j in range(18):
        taskunitinds.append(Huse * i + H + j)
taskunitinds = np.array(taskunitinds).astype(int)

hiddeninds = []
for i in range(12):
    for j in range(H):
        hiddeninds.append(Huse * i + j)
hiddeninds = np.array(hiddeninds).astype(int)

datainputinds = []
for i in range(H):
    for j in range(18):
        datainputinds.append(Dinuse * i + j)
taskinputinds = []
for i in range(H):
    for j in range(18):
        taskinputinds.append(Dinuse * i + 18 + j)
datainputinds = np.array(datainputinds).astype(int)
taskinputinds = np.array(taskinputinds).astype(int)

model0_w = np.copy(model.model.fc1.weight.data.numpy())
model0_w2 = np.copy(model.model.fc2.weight.data.numpy())

lstore_train = np.zeros(NumIters + 1)
lstore_test = np.zeros(NumIters + 1)

lstore_train2 = np.zeros(NumIters + 1)
lstore_test2 = np.zeros(NumIters + 1)

for t in range(NumIters + 1):
    optimizer.zero_grad()
    optimizer2.zero_grad()

    y_pred, hidden = model(xtrain)
    y_pred2, hidden2 = model(xtest)
    y_pred3, hidden3 = model2(xtrain)
    y_pred4, hidden4 = model2(xtest)
    loss = loss_fn(y_pred, ytrain)
    loss_sc = loss.item() ** 0.5
    losstest = loss_fn(y_pred2, ytest)
    lstore_train[t] = loss.item()
    lstore_test[t] = losstest.item()
    loss2 = loss_fn(y_pred3, ytrain)
    loss_sc2 = loss2.item() ** 0.5
    losstest2 = loss_fn(y_pred4, ytest)
    lstore_train2[t] = loss2.item()
    lstore_test2[t] = losstest2.item()
    # Print loss
    if t % PrintFreq == 0:
        print(t, loss.item(), loss2.item())

    # Greatly simplified storage. Only store loss, hidden. Ignore all NTK analysis
    if t % PlotFreq == 0:  # Store data for plot, in this case we need hidden activations
        with torch.no_grad():
            hstore[t // PlotFreq, :, :] = hidden.data.numpy()
            lstore[t // PlotFreq] = loss.item()
            hstore2[t // PlotFreq, :, :] = hidden3.data.numpy()
            lstore2[t // PlotFreq] = loss2.item()

    if t==0:
        NTKtest = np.zeros((Dout, Ntest, numparams))
        for j in range(Dout):
            yout2 = torch.sum(y_pred2, 0)
            yout2[j].backward(retain_graph=True)
            model.reduce_batch()
            test = [p.bgrad for p in model.parameters()]
            for i in range(Ntest):
                graduse = np.empty(0)
                kuse = 0
                for k in model.parameters():
                    if k.dim() == 2:  # only count the weights, not biases
                        graduse = np.append(graduse, test[kuse][i].data.numpy().flatten())
                    kuse += 1
                NTKtest[j, i, :] = graduse
            optimizer.zero_grad()
        NTKtest2 = np.zeros((Dout, Ntest, numparams))
        for j in range(Dout):
            yout4 = torch.sum(y_pred4, 0)
            yout4[j].backward(retain_graph=True)
            model2.reduce_batch()
            test = [p.bgrad for p in model2.parameters()]
            for i in range(Ntest):
                graduse = np.empty(0)
                kuse = 0
                for k in model2.parameters():
                    if k.dim() == 2:  # only count the weights, not biases
                        graduse = np.append(graduse, test[kuse][i].data.numpy().flatten())
                    kuse += 1
                NTKtest2[j, i, :] = graduse
            optimizer2.zero_grad()

    optimizer.zero_grad()
    optimizer2.zero_grad()
    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    loss2.backward()
    model.reduce_batch()
    model2.reduce_batch()

    if t==0:
        NTKtrain = np.zeros((1, N, numparams))
        test = [p.bgrad for p in
                model.parameters()]  # using last backward e.g. loss e.g. should already inlude dl/dy?
        for j in range(1):
            for i in range(N):
                graduse = np.empty(0)
                kuse = 0
                for k in model.parameters():
                    if k.dim() == 2:  # only count the weights, not biases
                        graduse = np.append(graduse, test[kuse][i].data.numpy().flatten())
                    kuse += 1
                NTKtrain[j, i, :] = graduse
        NTK = np.einsum('kjl, kil ->ijk', NTKtrain[:, :, :], NTKtest[:, :, :])

        NTKtrain2 = np.zeros((1, N, numparams))
        test = [p.bgrad for p in
                model2.parameters()]  # using last backward e.g. loss e.g. should already inlude dl/dy?
        for j in range(1):
            for i in range(N):
                graduse = np.empty(0)
                kuse = 0
                for k in model2.parameters():
                    if k.dim() == 2:  # only count the weights, not biases
                        graduse = np.append(graduse, test[kuse][i].data.numpy().flatten())
                    kuse += 1
                NTKtrain2[j, i, :] = graduse
        NTK2 = np.einsum('kjl, kil ->ijk', NTKtrain2[:, :, :], NTKtest2[:, :, :])

    # Update parameters according to chosen optimizer
    optimizer.step()
    optimizer2.step()
    with torch.no_grad():
        model.model.fc1.bias[:] = torch.ones(H) * bias_init
        model.model.fc2.bias[:] = torch.ones(Dout) * bias_init
        model2.model.fc1.bias[:] = torch.ones(H) * bias_init
        model2.model.fc2.bias[:] = torch.ones(Dout) * bias_init
    model.reassign_params()  # required for multi
    model2.reassign_params()  # required for multi

    if t == 0:
        NTKevals = np.zeros((Dout, N))
        NTKevecs = np.zeros((Dout, N, numeigvec))
        NTK0 = np.copy(NTK)
        NTKtrain0 = np.copy(NTKtrain)
        NTKtest0 = np.copy(NTKtest)
        hidden_0 = np.copy(hidden.data.numpy())
        hidden2_0 = np.copy(hidden2.data.numpy())
        with torch.no_grad():
            for k in range(Dout):
                eigvals, eigvecs = np.linalg.eig(NTK[:, :, k].transpose())
                NTKevals[k,:] = eigvals
                NTKevecs[k,:,:] = eigvecs[:, :numeigvec]
        NTKevals2 = np.zeros((Dout, N))
        NTKevecs2 = np.zeros((Dout, N, numeigvec))
        NTK02 = np.copy(NTK2)
        NTKtrain02 = np.copy(NTKtrain2)
        NTKtest02 = np.copy(NTKtest2)
        hidden_02 = np.copy(hidden3.data.numpy())
        hidden2_02 = np.copy(hidden4.data.numpy())
        with torch.no_grad():
            for k in range(Dout):
                eigvals2, eigvecs2 = np.linalg.eig(NTK2[:, :, k].transpose())
                NTKevals2[k, :] = eigvals2
                NTKevecs2[k, :, :] = eigvecs2[:, :numeigvec]

ytest_pred, hidden = model(xtest)
losstest = loss_fn(ytest_pred, ytest)
print(losstest.item())
ytest_pred2, hidden2 = model2(xtest)
losstest2 = loss_fn(ytest_pred2, ytest)
print(losstest2.item())
plt.plot(lstore_train)
plt.plot(lstore_test)
plt.plot(lstore_train2)
plt.plot(lstore_test2)
plt.title('Standard/High Initialization')
plt.legend(['Train Loss (Std - All)', 'Test Loss (Std - All)', 'Train Loss (Large - All)', 'Test Loss (Large - All)'])
plt.show()

plt.semilogy(lstore_test)
plt.semilogy(lstore_test2)
plt.legend(['Std','large'])
plt.title('Test lossees (generalization)')
plt.show()

# Do MPHATE analysis
# Over both hstore, hstore2

doMPHATE = False
if doMPHATE: # cut the 3D mphate plots. They take longer, aren't used in actual paper
    import m_phate

    print('Beginning M-PHATE for Std Init')
    if Ningroup==4:# Try breaking up by both task and input activation
        havgstore2 = np.zeros((numplot, len(taskids), len(np.unique(xactstore[:N])), H))
        numhstore2 = np.zeros((numplot, len(taskids), len(np.unique(xactstore[:N]))))
        for i in range(numplot):
            for j in range(N):
                havgstore2[i, int(tstore[j]), int(xactstore[j]), :] += hstore[i, j, :]
                numhstore2[i, int(tstore[j]), int(xactstore[j])] += 1
            for j in range(len(taskids)):
                for k in range(len(np.unique(xactstore[:N]))):
                    if numhstore2[i,j,k]>0:
                        havgstore2[i, j, k, :] /= numhstore2[i, j, k]
        # Eliminate 0s - there is actually only 36 (4 blocks of 9x9)... rest are 0s, can't have 0s in MPHATE
        havgstore2dense = np.zeros((numplot, 12, 3, H))
        havgstore2dense[:, :3, :, :] = havgstore2[:, :3, :3, :]
        havgstore2dense[:, 3:6, :, :] = havgstore2[:, 3:6, 3:6, :]
        havgstore2dense[:, 6:9, :, :] = havgstore2[:, 6:9, 6:9, :]
        havgstore2dense[:, 9:12, :, :] = havgstore2[:, 9:12, 9:12, :]
        # Reduce dimensionality
        havgstore2 = havgstore2dense.reshape((numplot, 12*3, H))
        # Get MPHATE
        m_phate_op22 = m_phate.M_PHATE()
        m_phate_data22 = m_phate_op22.fit_transform(havgstore2)
        #
        import scprep

        time22 = np.repeat(np.arange(numplot), 12 * 3)
        scprep.plot.scatter2d(m_phate_data22, c=time22, ticks=False, label_prefix="M-PHATE")

        # Next try breaking up only by input activation (not task)
        havgstore3 = np.zeros((numplot, len(np.unique(xactstore[:N])), H))
        numhstore3 = np.zeros((numplot, len(np.unique(xactstore[:N]))))
        for i in range(numplot):
            for j in range(N):
                havgstore3[i, int(xactstore[j]), :] += hstore[i, j, :]
                numhstore3[i, int(xactstore[j])] += 1
            for j in range(len(np.unique(xactstore[:N]))):
                havgstore3[i, j, :] /= numhstore3[i, j]
        # Get MPHATE
        m_phate_op23 = m_phate.M_PHATE()
        m_phate_data23 = m_phate_op23.fit_transform(havgstore3)
        #
        time23 = np.repeat(np.arange(numplot), len(np.unique(xactstore[:N])))
        scprep.plot.scatter2d(m_phate_data23, c=time23, ticks=False, label_prefix="M-PHATE")

        # New - also try breaking up by output!
        havgstore4 = np.zeros((numplot, len(np.unique(outstore[:N])), H))
        numhstore4 = np.zeros((numplot, len(np.unique(outstore[:N]))))
        for i in range(numplot):
            for j in range(N):
                havgstore4[i, int(outstore[j]), :] += hstore[i, j, :]
                numhstore4[i, int(outstore[j])] += 1
            for j in range(len(np.unique(outstore[:N]))):
                havgstore4[i, j, :] /= numhstore4[i, j]
        # Get MPHATE
        m_phate_op24 = m_phate.M_PHATE()
        m_phate_data24 = m_phate_op24.fit_transform(havgstore4)
        #
        time24 = np.repeat(np.arange(numplot), len(np.unique(outstore[:N])))
        scprep.plot.scatter2d(m_phate_data24, c=time24, ticks=False, label_prefix="M-PHATE")

        havgstore = np.zeros((numplot, len(taskids), H))
        numhstore = np.zeros((numplot, len(taskids)))
        for i in range(numplot):
            for j in range(N):
                havgstore[i, int(tstore[j]), :] += hstore[i, j, :]
                numhstore[i, int(tstore[j])] += 1
            for j in range(len(taskids)):
                if numhstore[i,j]>0:
                    havgstore[i, j, :] /= numhstore[i, j]
                else:
                    havgstore[i,j,:] = 0
        havgstore = havgstore[:,:12,:]
        m_phate_op2 = m_phate.M_PHATE()
        m_phate_data2 = m_phate_op2.fit_transform(havgstore)

    import matplotlib.pyplot as plt

    import scprep

    time2 = np.repeat(np.arange(numplot), 12)

    scprep.plot.scatter2d(m_phate_data2, c=time2, ticks=False, label_prefix="M-PHATE")

    # Repeat, but with the other data set...
    print('Beginning M-PHATE for Large Init')

    if Ningroup==4:# Try breaking up by both task and input activation
        havgstore2_2 = np.zeros((numplot, len(taskids), len(np.unique(xactstore[:N])), H))
        numhstore2_2 = np.zeros((numplot, len(taskids), len(np.unique(xactstore[:N]))))
        for i in range(numplot):
            for j in range(N):
                havgstore2_2[i, int(tstore[j]), int(xactstore[j]), :] += hstore2[i, j, :]
                numhstore2_2[i, int(tstore[j]), int(xactstore[j])] += 1
            for j in range(len(taskids)):
                for k in range(len(np.unique(xactstore[:N]))):
                    if numhstore2_2[i,j,k]>0:
                        havgstore2_2[i, j, k, :] /= numhstore2_2[i, j, k]
        # Eliminate 0s - there is actually only 36 (4 blocks of 9x9)... rest are 0s, can't have 0s in MPHATE
        havgstore2dense_2 = np.zeros((numplot, 12, 3, H))
        havgstore2dense_2[:, :3, :, :] = havgstore2_2[:, :3, :3, :]
        havgstore2dense_2[:, 3:6, :, :] = havgstore2_2[:, 3:6, 3:6, :]
        havgstore2dense_2[:, 6:9, :, :] = havgstore2_2[:, 6:9, 6:9, :]
        havgstore2dense_2[:, 9:12, :, :] = havgstore2_2[:, 9:12, 9:12, :]
        # Reduce dimensionality
        havgstore2_2 = havgstore2dense_2.reshape((numplot, 12*3, H))
        # Get MPHATE
        m_phate2_op22 = m_phate.M_PHATE()
        m_phate2_data22 = m_phate2_op22.fit_transform(havgstore2_2)
        #
        import scprep

        time22 = np.repeat(np.arange(numplot), 12 * 3)
        scprep.plot.scatter2d(m_phate2_data22, c=time22, ticks=False, label_prefix="M-PHATE")

        # Next try breaking up only by input activation (not task)

        havgstore3_2 = np.zeros((numplot, len(np.unique(xactstore[:N])), H))
        numhstore3_2 = np.zeros((numplot, len(np.unique(xactstore[:N]))))
        for i in range(numplot):
            for j in range(N):
                havgstore3_2[i, int(xactstore[j]), :] += hstore2[i, j, :]
                numhstore3_2[i, int(xactstore[j])] += 1
            for j in range(len(np.unique(xactstore[:N]))):
                havgstore3_2[i, j, :] /= numhstore3_2[i, j]
        # Get MPHATE
        m_phate2_op23 = m_phate.M_PHATE()
        m_phate2_data23 = m_phate2_op23.fit_transform(havgstore3_2)
        #
        time23 = np.repeat(np.arange(numplot), len(np.unique(xactstore[:N])))
        scprep.plot.scatter2d(m_phate2_data23, c=time23, ticks=False, label_prefix="M-PHATE")

        # New - also try breaking up by output!
        havgstore4_2 = np.zeros((numplot, len(np.unique(outstore[:N])), H))
        numhstore4_2 = np.zeros((numplot, len(np.unique(outstore[:N]))))
        for i in range(numplot):
            for j in range(N):
                havgstore4_2[i, int(outstore[j]), :] += hstore2[i, j, :]
                numhstore4_2[i, int(outstore[j])] += 1
            for j in range(len(np.unique(outstore[:N]))):
                havgstore4_2[i, j, :] /= numhstore4_2[i, j]
        # Get MPHATE
        m_phate2_op24 = m_phate.M_PHATE()
        m_phate2_data24 = m_phate2_op24.fit_transform(havgstore4_2)
        #
        time24 = np.repeat(np.arange(numplot), len(np.unique(outstore[:N])))
        scprep.plot.scatter2d(m_phate2_data24, c=time24, ticks=False, label_prefix="M-PHATE")

        havgstore_2 = np.zeros((numplot, len(taskids), H))
        numhstore_2 = np.zeros((numplot, len(taskids)))
        for i in range(numplot):
            for j in range(N):
                havgstore_2[i, int(tstore[j]), :] += hstore2[i, j, :]
                numhstore_2[i, int(tstore[j])] += 1
            for j in range(len(taskids)):
                if numhstore_2[i,j]>0:
                    havgstore_2[i, j, :] /= numhstore_2[i, j]
                else:
                    havgstore_2[i,j,:] = 0
        havgstore_2 = havgstore_2[:,:12,:]
        m_phate2_op2 = m_phate.M_PHATE()
        m_phate2_data2 = m_phate2_op2.fit_transform(havgstore_2)

    import matplotlib.pyplot as plt

    import scprep

    time2 = np.repeat(np.arange(numplot), 12)

    scprep.plot.scatter2d(m_phate2_data2, c=time2, ticks=False, label_prefix="M-PHATE")

# Save / Load / Analyze
#
import pickle
SaveDat = False
if SaveDat:
    # 2 variants: The lrtune (LR), and the extended time (ET)
    # lrtune first: done 123
    StdAllTrain11LRRep1 = f'data\Pickle\CohenStdAllTrain11LRRep1.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep1)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep1.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep1 = f'data\Pickle\CohenLargeAllTrain11LRRep1.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep1)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep1.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Rep 2

    StdAllTrain11LRRep2 = f'data\Pickle\CohenStdAllTrain11LRRep2.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep2)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep2.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep2 = f'data\Pickle\CohenLargeAllTrain11LRRep2.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep2)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep2.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Rep3

    StdAllTrain11LRRep3 = f'data\Pickle\CohenStdAllTrain11LRRep3.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep3)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep3.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep3 = f'data\Pickle\CohenLargeAllTrain11LRRep3.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep3)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep3.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Save extended time
    # Done: 123
    StdAllTrain11ETRep1 = f'data\Pickle\CohenStdAllTrain11ETRep1.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep1)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep1.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep1 = f'data\Pickle\CohenLargeAllTrain11ETRep1.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep1)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep1.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Rep 2
    StdAllTrain11ETRep2 = f'data\Pickle\CohenStdAllTrain11ETRep2.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep2)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep2.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep2 = f'data\Pickle\CohenLargeAllTrain11ETRep2.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep2)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep2.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Rep 3
    StdAllTrain11ETRep3 = f'data\Pickle\CohenStdAllTrain11ETRep3.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep3)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep3.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep3 = f'data\Pickle\CohenLargeAllTrain11ETRep3.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep3)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep3.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # On to Fine Tuning:
    # For these, no mphate data, swapping datasets to multi/multi
    # lrtune first: done 123
    StdAllTune11LRRep1 = f'data\Pickle\CohenStdAllTune11LRRep1.pt'
    torch.save(model.state_dict(), StdAllTune11LRRep1)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11LRRep1.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11LRRep1 = f'data\Pickle\CohenLargeAllTune11LRRep1.pt'
    torch.save(model2.state_dict(), LargeAllTune11LRRep1)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2,  NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11LRRep1.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Rep 2
    StdAllTune11LRRep2 = f'data\Pickle\CohenStdAllTune11LRRep2.pt'
    torch.save(model.state_dict(), StdAllTune11LRRep2)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11LRRep2.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11LRRep2 = f'data\Pickle\CohenLargeAllTune11LRRep2.pt'
    torch.save(model2.state_dict(), LargeAllTune11LRRep2)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11LRRep2.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Rep 3
    StdAllTune11LRRep3 = f'data\Pickle\CohenStdAllTune11LRRep3.pt'
    torch.save(model.state_dict(), StdAllTune11LRRep3)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11LRRep3.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11LRRep3 = f'data\Pickle\CohenLargeAllTune11LRRep3.pt'
    torch.save(model2.state_dict(), LargeAllTune11LRRep3)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11LRRep3.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Next up, fine-tune extended time.
    # extend time: done 123
    StdAllTune11ETRep1 = f'data\Pickle\CohenStdAllTune11ETRep1.pt'
    torch.save(model.state_dict(), StdAllTune11ETRep1)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11ETRep1.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11ETRep1 = f'data\Pickle\CohenLargeAllTune11ETRep1.pt'
    torch.save(model2.state_dict(), LargeAllTune11ETRep1)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11ETRep1.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Rep 2
    StdAllTune11ETRep2 = f'data\Pickle\CohenStdAllTune11ETRep2.pt'
    torch.save(model.state_dict(), StdAllTune11ETRep2)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11ETRep2.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11ETRep2 = f'data\Pickle\CohenLargeAllTune11ETRep2.pt'
    torch.save(model2.state_dict(), LargeAllTune11ETRep2)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11ETRep2.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Rep 3
    StdAllTune11ETRep3 = f'data\Pickle\CohenStdAllTune11ETRep3.pt'
    torch.save(model.state_dict(), StdAllTune11ETRep3)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11ETRep3.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11ETRep3 = f'data\Pickle\CohenLargeAllTune11ETRep3.pt'
    torch.save(model2.state_dict(), LargeAllTune11ETRep3)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11ETRep3.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)


    ##### Extension: Upgrade from 3 to 10 iters, or an additional +7. Base only, no MPHATE
    ##### Some of the later #s are already used for MPHATE, so I am upgrading 4 to 14, 5 to 15 etc

    ### LR Tuned - Base
    # Rep 4
    StdAllTrain11LRRep14 = f'data\Pickle\CohenStdAllTrain11LRRep14.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep14)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep14.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep14 = f'data\Pickle\CohenLargeAllTrain11LRRep14.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep14)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep14.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 5
    StdAllTrain11LRRep15 = f'data\Pickle\CohenStdAllTrain11LRRep15.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep15)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep15.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep15 = f'data\Pickle\CohenLargeAllTrain11LRRep15.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep15)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep15.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 6
    StdAllTrain11LRRep16 = f'data\Pickle\CohenStdAllTrain11LRRep16.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep16)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep16.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep16 = f'data\Pickle\CohenLargeAllTrain11LRRep16.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep16)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep16.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 7
    StdAllTrain11LRRep17 = f'data\Pickle\CohenStdAllTrain11LRRep17.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep17)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep17.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep17 = f'data\Pickle\CohenLargeAllTrain11LRRep17.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep17)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep17.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 8
    StdAllTrain11LRRep18 = f'data\Pickle\CohenStdAllTrain11LRRep18.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep18)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep18.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep18 = f'data\Pickle\CohenLargeAllTrain11LRRep18.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep18)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep18.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 9
    StdAllTrain11LRRep19 = f'data\Pickle\CohenStdAllTrain11LRRep19.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep19)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep19.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep19 = f'data\Pickle\CohenLargeAllTrain11LRRep19.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep19)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep19.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 10
    StdAllTrain11LRRep20 = f'data\Pickle\CohenStdAllTrain11LRRep20.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep20)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep20.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep20 = f'data\Pickle\CohenLargeAllTrain11LRRep20.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep20)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep20.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    ### ET - Base
    # Rep 4
    StdAllTrain11ETRep14 = f'data\Pickle\CohenStdAllTrain11ETRep14.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep14)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep14.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep14 = f'data\Pickle\CohenLargeAllTrain11ETRep14.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep14)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep14.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 5
    StdAllTrain11ETRep15 = f'data\Pickle\CohenStdAllTrain11ETRep15.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep15)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep15.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep15 = f'data\Pickle\CohenLargeAllTrain11ETRep15.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep15)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep15.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 6
    StdAllTrain11ETRep16 = f'data\Pickle\CohenStdAllTrain11ETRep16.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep16)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep16.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep16 = f'data\Pickle\CohenLargeAllTrain11ETRep16.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep16)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep16.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 7
    StdAllTrain11ETRep17 = f'data\Pickle\CohenStdAllTrain11ETRep17.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep17)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep17.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep17 = f'data\Pickle\CohenLargeAllTrain11ETRep17.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep17)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep17.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 8
    StdAllTrain11ETRep18 = f'data\Pickle\CohenStdAllTrain11ETRep18.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep18)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep18.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep18 = f'data\Pickle\CohenLargeAllTrain11ETRep18.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep18)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep18.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 9
    StdAllTrain11ETRep19 = f'data\Pickle\CohenStdAllTrain11ETRep19.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep19)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep19.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep19 = f'data\Pickle\CohenLargeAllTrain11ETRep19.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep19)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep19.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 10
    StdAllTrain11ETRep20 = f'data\Pickle\CohenStdAllTrain11ETRep20.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep20)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep20.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep20 = f'data\Pickle\CohenLargeAllTrain11ETRep20.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep20)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep20.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    ### LR Tuned - Fine Tuning
    # Rep 4
    StdAllTune11LRRep14 = f'data\Pickle\CohenStdAllTune11LRRep14.pt'
    torch.save(model.state_dict(), StdAllTune11LRRep14)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11LRRep14.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11LRRep14 = f'data\Pickle\CohenLargeAllTune11LRRep14.pt'
    torch.save(model2.state_dict(), LargeAllTune11LRRep14)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11LRRep14.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 5
    StdAllTune11LRRep15 = f'data\Pickle\CohenStdAllTune11LRRep15.pt'
    torch.save(model.state_dict(), StdAllTune11LRRep15)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11LRRep15.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11LRRep15 = f'data\Pickle\CohenLargeAllTune11LRRep15.pt'
    torch.save(model2.state_dict(), LargeAllTune11LRRep15)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11LRRep15.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 6
    StdAllTune11LRRep16 = f'data\Pickle\CohenStdAllTune11LRRep16.pt'
    torch.save(model.state_dict(), StdAllTune11LRRep16)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11LRRep16.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11LRRep16 = f'data\Pickle\CohenLargeAllTune11LRRep16.pt'
    torch.save(model2.state_dict(), LargeAllTune11LRRep16)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11LRRep16.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 7
    StdAllTune11LRRep17 = f'data\Pickle\CohenStdAllTune11LRRep17.pt'
    torch.save(model.state_dict(), StdAllTune11LRRep17)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11LRRep17.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11LRRep17 = f'data\Pickle\CohenLargeAllTune11LRRep17.pt'
    torch.save(model2.state_dict(), LargeAllTune11LRRep17)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11LRRep17.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 8
    StdAllTune11LRRep18 = f'data\Pickle\CohenStdAllTune11LRRep18.pt'
    torch.save(model.state_dict(), StdAllTune11LRRep18)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11LRRep18.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11LRRep18 = f'data\Pickle\CohenLargeAllTune11LRRep18.pt'
    torch.save(model2.state_dict(), LargeAllTune11LRRep18)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11LRRep18.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 9
    StdAllTune11LRRep19 = f'data\Pickle\CohenStdAllTune11LRRep19.pt'
    torch.save(model.state_dict(), StdAllTune11LRRep19)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11LRRep19.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11LRRep19 = f'data\Pickle\CohenLargeAllTune11LRRep19.pt'
    torch.save(model2.state_dict(), LargeAllTune11LRRep19)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11LRRep19.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 10
    StdAllTune11LRRep20 = f'data\Pickle\CohenStdAllTune11LRRep20.pt'
    torch.save(model.state_dict(), StdAllTune11LRRep20)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11LRRep20.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11LRRep20 = f'data\Pickle\CohenLargeAllTune11LRRep20.pt'
    torch.save(model2.state_dict(), LargeAllTune11LRRep20)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11LRRep20.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    ### ET - Fine Tuning
    # Rep 4
    StdAllTune11ETRep14 = f'data\Pickle\CohenStdAllTune11ETRep14.pt'
    torch.save(model.state_dict(), StdAllTune11ETRep14)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11ETRep14.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11ETRep14 = f'data\Pickle\CohenLargeAllTune11ETRep14.pt'
    torch.save(model2.state_dict(), LargeAllTune11ETRep14)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11ETRep14.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 5
    StdAllTune11ETRep15 = f'data\Pickle\CohenStdAllTune11ETRep15.pt'
    torch.save(model.state_dict(), StdAllTune11ETRep15)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11ETRep15.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11ETRep15 = f'data\Pickle\CohenLargeAllTune11ETRep15.pt'
    torch.save(model2.state_dict(), LargeAllTune11ETRep15)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11ETRep15.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 6
    StdAllTune11ETRep16 = f'data\Pickle\CohenStdAllTune11ETRep16.pt'
    torch.save(model.state_dict(), StdAllTune11ETRep16)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11ETRep16.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11ETRep16 = f'data\Pickle\CohenLargeAllTune11ETRep16.pt'
    torch.save(model2.state_dict(), LargeAllTune11ETRep16)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11ETRep16.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 7
    StdAllTune11ETRep17 = f'data\Pickle\CohenStdAllTune11ETRep17.pt'
    torch.save(model.state_dict(), StdAllTune11ETRep17)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11ETRep17.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11ETRep17 = f'data\Pickle\CohenLargeAllTune11ETRep17.pt'
    torch.save(model2.state_dict(), LargeAllTune11ETRep17)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11ETRep17.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 8
    StdAllTune11ETRep18 = f'data\Pickle\CohenStdAllTune11ETRep18.pt'
    torch.save(model.state_dict(), StdAllTune11ETRep18)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11ETRep18.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11ETRep18 = f'data\Pickle\CohenLargeAllTune11ETRep18.pt'
    torch.save(model2.state_dict(), LargeAllTune11ETRep18)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11ETRep18.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 9
    StdAllTune11ETRep19 = f'data\Pickle\CohenStdAllTune11ETRep19.pt'
    torch.save(model.state_dict(), StdAllTune11ETRep19)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11ETRep19.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11ETRep19 = f'data\Pickle\CohenLargeAllTune11ETRep19.pt'
    torch.save(model2.state_dict(), LargeAllTune11ETRep19)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11ETRep19.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Rep 10
    StdAllTune11ETRep20 = f'data\Pickle\CohenStdAllTune11ETRep20.pt'
    torch.save(model.state_dict(), StdAllTune11ETRep20)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK]
    with open(f'data\Pickle\CohenStdAllTune11ETRep20.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTune11ETRep20 = f'data\Pickle\CohenLargeAllTune11ETRep20.pt'
    torch.save(model2.state_dict(), LargeAllTune11ETRep20)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2, NTKevals2, NTK2]
    with open(f'data\Pickle\CohenLargeAllTune11ETRep20.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)


    # Given that all have been run, go ahead and do analyses
    # Recreate original NTK evec plots, MPHATE plots
    # Plot exemplar Train/Test curves, one for each of LR and ET
    # Run analyses showing Training results in better single
    # Run analyses showing Finetuning results in better multi

    ### Load all
    # Load Train 1-3
    with open(f'data\Pickle\CohenStdAllTrain11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET3 = pickle.load(file)

    # Load Tuned 1-3
    with open(f'data\Pickle\CohenStdAllTune11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET3 = pickle.load(file)


    ### Generate results
    # Plot 1
    figure, axs = plt.subplots(1,2, figsize = (12,6))
    axs[0].semilogy(StdTrainLR1[1])
    axs[0].semilogy(LargeTrainLR1[1])
    axs[0].legend(['Std', 'Large'], fontsize=14)
    axs[1].semilogy(StdTuneLR1[1])
    axs[1].semilogy(LargeTuneLR1[1])
    axs[1].legend(['Std', 'Large'], fontsize=14)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Generalization Loss', fontsize=16)
    axs[0].set_title('Generalization - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Generalization Loss', fontsize=16)
    axs[1].set_title('Generalization - Multi-Task Tuning', fontsize=16)
    plt.show()
    # Plot 2
    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(StdTrainET1[1])
    axs[0].semilogy(LargeTrainET1[1])
    axs[0].legend(['Std', 'Large'], fontsize=14)
    axs[1].semilogy(StdTuneET1[1])
    axs[1].semilogy(LargeTuneET1[1])
    axs[1].legend(['Std', 'Large'], fontsize=14)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Generalization Loss', fontsize=16)
    axs[0].set_title('Generalization - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Generalization Loss', fontsize=16)
    axs[1].set_title('Generalization - Multi-Task Tuning', fontsize=16)
    plt.show()

    # Have another version that also includes train loss?
    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(StdTrainLR1[1])
    axs[0].semilogy(LargeTrainLR1[1])
    axs[0].semilogy(StdTrainLR1[0], c = '#1f77b4', linestyle = 'dotted')
    axs[0].semilogy(LargeTrainLR1[0], c = '#ff7f0e', linestyle = 'dotted')
    axs[0].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[1].semilogy(StdTuneLR1[1])
    axs[1].semilogy(LargeTuneLR1[1])
    axs[1].semilogy(StdTuneLR1[0], c = '#1f77b4', linestyle = 'dotted')
    axs[1].semilogy(LargeTuneLR1[0], c = '#ff7f0e', linestyle = 'dotted')
    axs[1].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Loss', fontsize=16)
    axs[0].set_title('Losses - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Loss', fontsize=16)
    axs[1].set_title('Losses - Multi-Task Tuning', fontsize=16)
    plt.show()

    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(StdTrainET1[1])
    axs[0].semilogy(LargeTrainET1[1])
    axs[0].semilogy(StdTrainET1[0], c = '#1f77b4', linestyle = 'dotted')
    axs[0].semilogy(LargeTrainET1[0], c = '#ff7f0e', linestyle = 'dotted')
    axs[0].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[1].semilogy(StdTuneET1[1])
    axs[1].semilogy(LargeTuneET1[1])
    axs[1].semilogy(StdTuneET1[0], c = '#1f77b4', linestyle = 'dotted')
    axs[1].semilogy(LargeTuneET1[0], c = '#ff7f0e', linestyle = 'dotted')
    axs[1].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Loss', fontsize=16)
    axs[0].set_title('Losses - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Loss', fontsize=16)
    axs[1].set_title('Losses - Multi-Task Tuning', fontsize=16)
    plt.show()

    # Analysis 1 - Training

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    silhouette_scores = np.zeros(12)
    datuse = np.zeros((12, 500, 9))
    datuse[0, :, :] = np.transpose(StdTrainET1[8][:, :, 0])
    datuse[1, :, :] = np.transpose(StdTrainET2[8][:, :, 0])
    datuse[2, :, :] = np.transpose(StdTrainET3[8][:, :, 0])
    datuse[3, :, :] = np.transpose(StdTrainLR1[8][:, :, 0])
    datuse[4, :, :] = np.transpose(StdTrainLR2[8][:, :, 0])
    datuse[5, :, :] = np.transpose(StdTrainLR3[8][:, :, 0])

    datuse[6, :, :] = np.transpose(LargeTrainET1[8][:, :, 0])
    datuse[7, :, :] = np.transpose(LargeTrainET2[8][:, :, 0])
    datuse[8, :, :] = np.transpose(LargeTrainET3[8][:, :, 0])
    datuse[9, :, :] = np.transpose(LargeTrainLR1[8][:, :, 0])
    datuse[10, :, :] = np.transpose(LargeTrainLR2[8][:, :, 0])
    datuse[11, :, :] = np.transpose(LargeTrainLR3[8][:, :, 0])

    for i in range(12):
        n_clusters = 12  # 4x3
        X = datuse[i, :, :]
        clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores[i] = silhouette_avg

    finaltestlosses_train = np.zeros(12)
    finaltestlosses_train[0] = StdTrainET1[1][-1]
    finaltestlosses_train[1] = StdTrainET2[1][-1]
    finaltestlosses_train[2] = StdTrainET3[1][-1]
    finaltestlosses_train[3] = StdTrainLR1[1][-1]
    finaltestlosses_train[4] = StdTrainLR2[1][-1]
    finaltestlosses_train[5] = StdTrainLR3[1][-1]
    finaltestlosses_train[6] = LargeTrainET1[1][-1]
    finaltestlosses_train[7] = LargeTrainET2[1][-1]
    finaltestlosses_train[8] = LargeTrainET3[1][-1]
    finaltestlosses_train[9] = LargeTrainLR1[1][-1]
    finaltestlosses_train[10] = LargeTrainLR2[1][-1]
    finaltestlosses_train[11] = LargeTrainLR3[1][-1]

    corr1 = np.corrcoef(silhouette_scores, finaltestlosses_train)[0, 1]
    silhouette_scores_1 = np.zeros(6)
    silhouette_scores_1[:3] = silhouette_scores[:3]
    silhouette_scores_1[3:6] = silhouette_scores[6:9]
    silhouette_scores_2 = np.zeros(6)
    silhouette_scores_2[:3] = silhouette_scores[3:6]
    silhouette_scores_2[3:6] = silhouette_scores[9:12]
    finaltestlosses_train_1 = np.zeros(6)
    finaltestlosses_train_1[:3] = finaltestlosses_train[:3]
    finaltestlosses_train_1[3:6] = finaltestlosses_train[6:9]
    finaltestlosses_train_2 = np.zeros(6)
    finaltestlosses_train_2[:3] = finaltestlosses_train[3:6]
    finaltestlosses_train_2[3:6] = finaltestlosses_train[9:12]
    corr1_1 = np.corrcoef(silhouette_scores_1, finaltestlosses_train_1)[0, 1]
    corr1_2 = np.corrcoef(silhouette_scores_2, finaltestlosses_train_2)[0, 1]

    # Analysis 2 - Tuning

    finaltestlosses_tune = np.zeros(12)
    finaltestlosses_tune[0] = StdTuneET1[1][-1]
    finaltestlosses_tune[1] = StdTuneET2[1][-1]
    finaltestlosses_tune[2] = StdTuneET3[1][-1]
    finaltestlosses_tune[3] = StdTuneLR1[1][-1]
    finaltestlosses_tune[4] = StdTuneLR2[1][-1]
    finaltestlosses_tune[5] = StdTuneLR3[1][-1]
    finaltestlosses_tune[6] = LargeTuneET1[1][-1]
    finaltestlosses_tune[7] = LargeTuneET2[1][-1]
    finaltestlosses_tune[8] = LargeTuneET3[1][-1]
    finaltestlosses_tune[9] = LargeTuneLR1[1][-1]
    finaltestlosses_tune[10] = LargeTuneLR2[1][-1]
    finaltestlosses_tune[11] = LargeTuneLR3[1][-1]

    corr2 = np.corrcoef(silhouette_scores, finaltestlosses_tune)[0, 1]
    finaltestlosses_tune_1 = np.zeros(6)
    finaltestlosses_tune_1[:3] = finaltestlosses_tune[:3]
    finaltestlosses_tune_1[3:6] = finaltestlosses_tune[6:9]
    finaltestlosses_tune_2 = np.zeros(6)
    finaltestlosses_tune_2[:3] = finaltestlosses_tune[3:6]
    finaltestlosses_tune_2[3:6] = finaltestlosses_tune[9:12]
    corr2_1 = np.corrcoef(silhouette_scores_1, finaltestlosses_tune_1)[0, 1]
    corr2_2 = np.corrcoef(silhouette_scores_2, finaltestlosses_tune_2)[0, 1]


    NTK_std_input = np.zeros((3,3,12))
    NTK_std_task = np.zeros((3,3,12))
    NTK_large_input = np.zeros((3,3,12))
    NTK_large_task = np.zeros((3,3,12))

    EMAeigs_std_input = np.zeros((9, N))
    EMAeigs_std_task = np.zeros((9, N))
    EMAeigs_large_input = np.zeros((9, N))
    EMAeigs_large_task = np.zeros((9, N))
    splits_std_task = np.concatenate([np.array([0]),np.cumsum(StdTrainET1[6][:(Din // 2)])]).astype(int)
    splits_std_input = np.concatenate([np.array([0]), np.cumsum(StdTrainET1[5][:(Din // 2)])]).astype(int)
    splits_large_task = np.concatenate([np.array([0]), np.cumsum(LargeTrainET1[6][:(Din // 2)])]).astype(int)
    splits_large_input = np.concatenate([np.array([0]), np.cumsum(LargeTrainET1[5][:(Din // 2)])]).astype(int)
    for i in range(3): # m
        for j in range(3): # g_2
            k = j*3 + i
            EMAeigs_std_input[k, :] = np.abs(StdTrainET1[8][k, StdTrainET1[3], 0])
            EMAeigs_std_task[k, :] = np.abs(StdTrainET1[8][k, StdTrainET1[2], 0])
            EMAeigs_large_input[k, :] = np.abs(LargeTrainET1[8][k, LargeTrainET1[3], 0])
            EMAeigs_large_task[k, :] = np.abs(LargeTrainET1[8][k, LargeTrainET1[2], 0])
            for l in range(12): # output
                NTK_std_input[j,i,l] = np.mean(EMAeigs_std_input[k,splits_std_input[l]:splits_std_input[l+1]])
                NTK_std_task[j,i,l] = np.mean(EMAeigs_std_task[k,splits_std_task[l]:splits_std_task[l+1]])
                NTK_large_input[j, i, l] = np.mean(EMAeigs_large_input[k, splits_large_input[l]:splits_large_input[l + 1]])
                NTK_large_task[j, i, l] = np.mean(EMAeigs_large_task[k, splits_large_task[l]:splits_large_task[l + 1]])


    NTK_std_task_plot = NTK_std_task.reshape(3,3*12)
    NTK_std_input_plot = NTK_std_input.reshape(3,3*12)
    NTK_large_task_plot = NTK_large_task.reshape(3,3*12)
    NTK_large_input_plot = NTK_large_input.reshape(3,3*12)

    fig, ax = plt.subplots(3, figsize = (10,10))
    fig.suptitle('NTK: 1st Eigenvalue\'s mean Eigenvector Components per Task, T = 0', fontsize=16)
    ax[0].matshow(NTK_std_task_plot[:, 0:12])
    ax[1].matshow(NTK_std_task_plot[:, 12:24])
    ax[2].matshow(NTK_std_task_plot[:, 24:])
    ax[0].set_title('$m = 1$', fontsize = 16)
    ax[1].set_title('$m = 2$', fontsize = 16)
    ax[2].set_title('$m = 3$', fontsize = 16)
    ax[0].set_ylabel('$g_2$', fontsize = 16)
    ax[0].set_yticks([0,1,2])
    ax[0].set_yticklabels(['1','2','3'])
    ax[1].set_ylabel('$g_2$', fontsize = 16)
    ax[1].set_yticks([0, 1, 2])
    ax[1].set_yticklabels(['1', '2', '3'])
    ax[2].set_ylabel('$g_2$', fontsize = 16)
    ax[2].set_yticks([0, 1, 2])
    ax[2].set_yticklabels(['1', '2', '3'])
    ax[0].set_xlabel('Task Group Index', fontsize=16)
    ax[1].set_xlabel('Task Group Index', fontsize=16)
    ax[2].set_xlabel('Task Group Index', fontsize=16)
    ax[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[2].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[0].set_xticks(np.arange(12))
    ax[0].set_xticklabels(['1', '2', '3','4','5','6','7','8','9','10','11','12'])
    ax[1].set_xticks(np.arange(12))
    ax[1].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    ax[2].set_xticks(np.arange(12))
    ax[2].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    plt.show()

    fig, ax = plt.subplots(3, figsize=(10, 10))
    fig.suptitle('NTK: 1st Eigenvalue\'s mean Eigenvector Components per Input, T = 0', fontsize=16)
    ax[0].matshow(NTK_std_input_plot[:, 0:12])
    ax[1].matshow(NTK_std_input_plot[:, 12:24])
    ax[2].matshow(NTK_std_input_plot[:, 24:])
    ax[0].set_title('$m = 1$', fontsize=16)
    ax[1].set_title('$m = 2$', fontsize=16)
    ax[2].set_title('$m = 3$', fontsize=16)
    ax[0].set_ylabel('$g_2$', fontsize=16)
    ax[0].set_yticks([0, 1, 2])
    ax[0].set_yticklabels(['1', '2', '3'])
    ax[1].set_ylabel('$g_2$', fontsize=16)
    ax[1].set_yticks([0, 1, 2])
    ax[1].set_yticklabels(['1', '2', '3'])
    ax[2].set_ylabel('$g_2$', fontsize=16)
    ax[2].set_yticks([0, 1, 2])
    ax[2].set_yticklabels(['1', '2', '3'])
    ax[0].set_xlabel('Input Group Index', fontsize=16)
    ax[1].set_xlabel('Input Group Index', fontsize=16)
    ax[2].set_xlabel('Input Group Index', fontsize=16)
    ax[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[2].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[0].set_xticks(np.arange(12))
    ax[0].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    ax[1].set_xticks(np.arange(12))
    ax[1].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    ax[2].set_xticks(np.arange(12))
    ax[2].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    plt.show()

    fig, ax = plt.subplots(3, figsize=(10, 10))
    fig.suptitle('NTK: 1st Eigenvalue\'s mean Eigenvector Components per Task, T = 0', fontsize=16)
    ax[0].matshow(NTK_large_task_plot[:, 0:12])
    ax[1].matshow(NTK_large_task_plot[:, 12:24])
    ax[2].matshow(NTK_large_task_plot[:, 24:])
    ax[0].set_title('$m = 1$', fontsize=16)
    ax[1].set_title('$m = 2$', fontsize=16)
    ax[2].set_title('$m = 3$', fontsize=16)
    ax[0].set_ylabel('$g_2$', fontsize=16)
    ax[0].set_yticks([0, 1, 2])
    ax[0].set_yticklabels(['1', '2', '3'])
    ax[1].set_ylabel('$g_2$', fontsize=16)
    ax[1].set_yticks([0, 1, 2])
    ax[1].set_yticklabels(['1', '2', '3'])
    ax[2].set_ylabel('$g_2$', fontsize=16)
    ax[2].set_yticks([0, 1, 2])
    ax[2].set_yticklabels(['1', '2', '3'])
    ax[0].set_xlabel('Task Group Index', fontsize=16)
    ax[1].set_xlabel('Task Group Index', fontsize=16)
    ax[2].set_xlabel('Task Group Index', fontsize=16)
    ax[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[2].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[0].set_xticks(np.arange(12))
    ax[0].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    ax[1].set_xticks(np.arange(12))
    ax[1].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    ax[2].set_xticks(np.arange(12))
    ax[2].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    plt.show()

    fig, ax = plt.subplots(3, figsize=(10, 10))
    fig.suptitle('NTK: 1st Eigenvalue\'s mean Eigenvector Components per Input, T = 0', fontsize=16)
    ax[0].matshow(NTK_large_input_plot[:, 0:12])
    ax[1].matshow(NTK_large_input_plot[:, 12:24])
    ax[2].matshow(NTK_large_input_plot[:, 24:])
    ax[0].set_title('$m = 1$', fontsize=16)
    ax[1].set_title('$m = 2$', fontsize=16)
    ax[2].set_title('$m = 3$', fontsize=16)
    ax[0].set_ylabel('$g_2$', fontsize=16)
    ax[0].set_yticks([0, 1, 2])
    ax[0].set_yticklabels(['1', '2', '3'])
    ax[1].set_ylabel('$g_2$', fontsize=16)
    ax[1].set_yticks([0, 1, 2])
    ax[1].set_yticklabels(['1', '2', '3'])
    ax[2].set_ylabel('$g_2$', fontsize=16)
    ax[2].set_yticks([0, 1, 2])
    ax[2].set_yticklabels(['1', '2', '3'])
    ax[0].set_xlabel('Input Group Index', fontsize=16)
    ax[1].set_xlabel('Input Group Index', fontsize=16)
    ax[2].set_xlabel('Input Group Index', fontsize=16)
    ax[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[2].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax[0].set_xticks(np.arange(12))
    ax[0].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    ax[1].set_xticks(np.arange(12))
    ax[1].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    ax[2].set_xticks(np.arange(12))
    ax[2].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    plt.show()




    # Rework of Fig 8 (now 9)
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('eNTK: 1st Eigenval/vec, organized by Task, T = 0', fontsize=15)
    # Reversed m, g_2
    plt.text(-1150, .6, '$g_2 = 1$', fontsize=16, rotation=0)
    plt.text(-450, .6, '$g_2 = 2$', fontsize=16, rotation=0)
    plt.text(200, .6, '$g_2 = 3$', fontsize=16, rotation=0)
    plt.text(-1550, .04, '$m = 3$', fontsize=16, rotation=90)
    plt.text(-1550, .25, '$m = 2$', fontsize=16, rotation=90)
    plt.text(-1550, .46, '$m = 1$', fontsize=16, rotation=90)
    EMAeigs = np.zeros((9, N))
    for k in range(9):
        EMAeigs[k, :] = np.abs(StdTrainET1[8][k, StdTrainET1[2], 0])
        # was [k//3][k%3]. Reversing swaps the outer x/y indices
        ax[k % 3][k // 3].plot(EMAeigs[k, :])
        ax[k % 3][k // 3].vlines(np.cumsum(StdTrainET1[6][:(Din // 2)]),
                                 np.min(StdTrainET1[8][k, StdTrainET1[2], 0]),
                                 np.max(np.abs(StdTrainET1[8][k, StdTrainET1[2], 0])))
        ax[k % 3][k // 3].set_title(f'eNTK Eigenvalue: {StdTrainET1[9][k, 0]:.3e}', fontsize=15)
        if k == 0:
            ax[k % 3][k // 3].set_xlabel('Sorted Index', fontsize=12)
            ax[k % 3][k // 3].set_ylabel('Eigenvector Component', fontsize=12)
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('eNTK: 1st Eigenval/vec, organized by Input, T = 0', fontsize=15)
    plt.text(-1150, .6, '$g_2 = 1$', fontsize=16, rotation=0)
    plt.text(-450, .6, '$g_2 = 2$', fontsize=16, rotation=0)
    plt.text(200, .6, '$g_2 = 3$', fontsize=16, rotation=0)
    plt.text(-1550, .04, '$m = 3$', fontsize=16, rotation=90)
    plt.text(-1550, .25, '$m = 2$', fontsize=16, rotation=90)
    plt.text(-1550, .46, '$m = 1$', fontsize=16, rotation=90)
    EMAeigs = np.zeros((9, N))
    for k in range(9):
        EMAeigs[k, :] = np.abs(StdTrainET1[8][k, StdTrainET1[3], 0])
        ax[k % 3][k // 3].plot(EMAeigs[k, :])
        ax[k % 3][k // 3].vlines(np.cumsum(StdTrainET1[5][:(Din // 2)]),
                                 np.min(StdTrainET1[8][k, StdTrainET1[3], 0]),
                                 np.max(np.abs(StdTrainET1[8][k, StdTrainET1[3], 0])))
        ax[k % 3][k // 3].set_title(f'eNTK Eigenvalue: {StdTrainET1[9][k, 0]:.3e}', fontsize=15)
        if k == 0:
            ax[k % 3][k // 3].set_xlabel('Sorted Index', fontsize=12)
            ax[k % 3][k // 3].set_ylabel('Eigenvector Component', fontsize=12)
    plt.show()

    # New figure - Appdx Fig showing unordered
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('eNTK: 1st Eigenval/vec, T = 0', fontsize = 15)
    # Reversed m, g_2
    plt.text(-1100, .5, '$g_2 = 1$', fontsize=16, rotation=0)
    plt.text(-450, .5, '$g_2 = 2$', fontsize=16, rotation=0)
    plt.text(200, .5, '$g_2 = 3$', fontsize=16, rotation=0)
    plt.text(-1500, .07, '$m = 3$', fontsize=16, rotation=90)
    plt.text(-1500, .24, '$m = 2$', fontsize=16, rotation=90)
    plt.text(-1500, .40, '$m = 1$', fontsize=16, rotation=90)
    EMAeigs = np.zeros((9, N))
    for k in range(9):
        EMAeigs[k, :] = np.abs(StdTrainET1[8][k, :, 0])
        # was [k//3][k%3]. Reversing swaps the outer x/y indices
        ax[k % 3][k // 3].plot(EMAeigs[k, :])
        ax[k % 3][k // 3].set_title(f'eNTK Eigenvalue: {StdTrainET1[9][k, 0]:.3e}', fontsize = 15)
        if k == 0:
            ax[k % 3][k // 3].set_xlabel('Index', fontsize = 12)
            ax[k % 3][k // 3].set_ylabel('Eigenvector Component', fontsize = 12)
    plt.show()

    # Rework of Appdx Fig 14 - Large model. Just replace Std w/ Large
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('eNTK: 1st Eigenval/vec, organized by Task, T = 0', fontsize=15)
    # Reversed m, g_2
    plt.text(-1100, .65, '$g_2 = 1$', fontsize=16, rotation=0)
    plt.text(-450, .65, '$g_2 = 2$', fontsize=16, rotation=0)
    plt.text(200, .65, '$g_2 = 3$', fontsize=16, rotation=0)
    plt.text(-1550, 0, '$m = 3$', fontsize=16, rotation=90)
    plt.text(-1550, .25, '$m = 2$', fontsize=16, rotation=90)
    plt.text(-1550, .5, '$m = 1$', fontsize=16, rotation=90)
    EMAeigs = np.zeros((9, N))
    for k in range(9):
        EMAeigs[k, :] = np.abs(LargeTrainET1[8][k, LargeTrainET1[2], 0])
        # was [k//3][k%3]. Reversing swaps the outer x/y indices
        ax[k % 3][k // 3].plot(EMAeigs[k, :])
        ax[k % 3][k // 3].vlines(np.cumsum(LargeTrainET1[6][:(Din // 2)]),
                                 np.min(LargeTrainET1[8][k, LargeTrainET1[2], 0]),
                                 np.max(np.abs(LargeTrainET1[8][k, LargeTrainET1[2], 0])))
        ax[k % 3][k // 3].set_title(f'eNTK Eigenvalue: {LargeTrainET1[9][k, 0]:.3e}', fontsize=15)
        if k == 0:
            ax[k % 3][k // 3].set_xlabel('Sorted Index', fontsize=12)
            ax[k % 3][k // 3].set_ylabel('Eigenvector Component', fontsize=12)
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('eNTK: 1st Eigenval/vec, organized by Input, T = 0', fontsize=15)
    plt.text(-1100, .65, '$g_2 = 1$', fontsize=16, rotation=0)
    plt.text(-450, .65, '$g_2 = 2$', fontsize=16, rotation=0)
    plt.text(200, .65, '$g_2 = 3$', fontsize=16, rotation=0)
    plt.text(-1550, 0, '$m = 3$', fontsize=16, rotation=90)
    plt.text(-1550, .25, '$m = 2$', fontsize=16, rotation=90)
    plt.text(-1550, .5, '$m = 1$', fontsize=16, rotation=90)
    EMAeigs = np.zeros((9, N))
    for k in range(9):
        EMAeigs[k, :] = np.abs(LargeTrainET1[8][k, LargeTrainET1[3], 0])
        ax[k % 3][k // 3].plot(EMAeigs[k, :])
        ax[k % 3][k // 3].vlines(np.cumsum(LargeTrainET1[5][:(Din // 2)]),
                                 np.min(LargeTrainET1[8][k, LargeTrainET1[3], 0]),
                                 np.max(np.abs(LargeTrainET1[8][k, LargeTrainET1[3], 0])))
        ax[k % 3][k // 3].set_title(f'eNTK Eigenvalue: {LargeTrainET1[9][k, 0]:.3e}', fontsize=15)
        if k == 0:
            ax[k % 3][k // 3].set_xlabel('Sorted Index', fontsize=12)
            ax[k % 3][k // 3].set_ylabel('Eigenvector Component', fontsize=12)
    plt.show()

    ## MPHATE
    # NOTES: tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
    #                NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    # So, mphate data is task[11] task by data [12] data[13] and output [14] ?

    # Ouch, MPHATE was wrong in one case...accidentally saved the std twice instead of std/large for task based
    # Construct LR4, ET4 (train only) just to get a correct MPHATE run, need for next plot too

    # Save Rep4, LR
    StdAllTrain11LRRep4 = f'data\Pickle\CohenStdAllTrain11LRRep4.pt'
    torch.save(model.state_dict(), StdAllTrain11LRRep4)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11LRRep4.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11LRRep4 = f'data\Pickle\CohenLargeAllTrain11LRRep4.pt'
    torch.save(model2.state_dict(), LargeAllTrain11LRRep4)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11LRRep4.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)
    # Save Rep4, ET
    StdAllTrain11ETRep4 = f'data\Pickle\CohenStdAllTrain11ETRep4.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep4)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep4.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep4 = f'data\Pickle\CohenLargeAllTrain11ETRep4.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep4)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep4.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)


    # Load Rep4, LR and ET
    with open(f'data\Pickle\CohenStdAllTrain11LRRep4.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR4 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep4.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR4 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep4.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET4 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep4.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET4 = pickle.load(file)


    import scprep

    # Input x Task (22,32 : 2,3)
    numplot = 101

    time22 = np.repeat(np.arange(numplot), 12 * 3)

    # Input (23,33 : 4,5)

    time23 = np.repeat(np.arange(numplot), 12)

    # Ouput (24,34: 6,7)

    time24 = np.repeat(np.arange(numplot), Dout)

    # Task (2,3: 8,9)

    time2 = np.repeat(np.arange(numplot), 12)

    # fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    scprep.plot.scatter2d(StdTrainET4[11], c=time2, ticks=False, label_prefix="M-PHATE", ax=ax[0][0])
    scprep.plot.scatter2d(LargeTrainET4[11], c=time2, ticks=False, label_prefix="M-PHATE", ax=ax[1][0])
    scprep.plot.scatter2d(StdTrainET4[13], c=time23, ticks=False, label_prefix="M-PHATE", ax=ax[0][1])
    scprep.plot.scatter2d(LargeTrainET4[13], c=time23, ticks=False, label_prefix="M-PHATE", ax=ax[1][1])
    scprep.plot.scatter2d(StdTrainET4[12], c=time22, ticks=False, label_prefix="M-PHATE", ax=ax[0][2])
    scprep.plot.scatter2d(LargeTrainET4[12], c=time22, ticks=False, label_prefix="M-PHATE", ax=ax[1][2])
    # Optional : output as well
    scprep.plot.scatter2d(StdTrainET4[14], c=time24, ticks=False, label_prefix="M-PHATE", ax=ax[0][3])
    scprep.plot.scatter2d(LargeTrainET4[14], c=time24, ticks=False, label_prefix="M-PHATE", ax=ax[1][3])
    # Text below assumes we are doing the output as well version - else numbers will probably change
    plt.text(-.5, .095, 'Std Init', fontsize=16, rotation=90)
    plt.text(-.5, -.02, 'Large Init', fontsize=16, rotation=90)
    plt.text(-.005, .16, 'Output', fontsize=16, rotation=0)
    plt.text(-.155, .16, 'Task by Inputs', fontsize=16, rotation=0)
    plt.text(-.285, .16, 'Inputs', fontsize=16, rotation=0)
    plt.text(-.425, .16, 'Task', fontsize=16, rotation=0)
    # He also wants text for color bar. Training %?
    plt.text(-.36, .045, 'Training %', fontsize=12, rotation=0)
    plt.text(-.22, .045, 'Training %', fontsize=12, rotation=0)
    plt.text(-.08, .045, 'Training %', fontsize=12, rotation=0)
    plt.text(.06, .045, 'Training %', fontsize=12, rotation=0)
    plt.text(-.36, .155, 'Training %', fontsize=12, rotation=0)
    plt.text(-.22, .155, 'Training %', fontsize=12, rotation=0)
    plt.text(-.08, .155, 'Training %', fontsize=12, rotation=0)
    plt.text(.06, .155, 'Training %', fontsize=12, rotation=0)

    x00 = ax[0][0].get_xlim()
    x10 = ax[1][0].get_xlim()
    x01 = ax[0][1].get_xlim()
    x11 = ax[1][1].get_xlim()
    x02 = ax[0][2].get_xlim()
    x12 = ax[1][2].get_xlim()
    x03 = ax[0][3].get_xlim()
    x13 = ax[1][3].get_xlim()

    y00 = ax[0][0].get_ylim()
    y10 = ax[1][0].get_ylim()
    y01 = ax[0][1].get_ylim()
    y11 = ax[1][1].get_ylim()
    y02 = ax[0][2].get_ylim()
    y12 = ax[1][2].get_ylim()
    y03 = ax[0][3].get_ylim()
    y13 = ax[1][3].get_ylim()

    # Optional - add in connecting lines
    MPHATE_Connect = True
    if MPHATE_Connect:
        for i in range(12):
            ax[0][0].plot(StdTrainET4[11][i:-1:12, 0], StdTrainET4[11][i:-1:12, 1], color = 'blue', alpha = .1)
            ax[1][0].plot(LargeTrainET4[11][i:-1:12, 0], LargeTrainET4[11][i:-1:12, 1], color='blue', alpha=.1)
            ax[0][1].plot(StdTrainET4[13][i:-1:12, 0], StdTrainET4[13][i:-1:12, 1], color='blue', alpha=.1)
            ax[1][1].plot(LargeTrainET4[13][i:-1:12, 0], LargeTrainET4[13][i:-1:12, 1], color='blue', alpha=.1)
        for i in range(12*3):
            ax[0][2].plot(StdTrainET4[12][i:-1:12*3, 0], StdTrainET4[12][i:-1:12*3, 1], color='blue', alpha=.1)
            ax[1][2].plot(LargeTrainET4[12][i:-1:12*3, 0], LargeTrainET4[12][i:-1:12*3, 1], color='blue', alpha=.1)
        for i in range(9):
            ax[0][3].plot(StdTrainET4[14][i:-1:9, 0], StdTrainET4[14][i:-1:9, 1], color='blue', alpha=.1)
            ax[1][3].plot(LargeTrainET4[14][i:-1:9, 0], LargeTrainET4[14][i:-1:9, 1], color='blue', alpha=.1)

        ax[0][0].set_xlim(x00)
        ax[1][0].set_xlim(x10)
        ax[0][1].set_xlim(x01)
        ax[1][1].set_xlim(x11)
        ax[0][2].set_xlim(x02)
        ax[1][2].set_xlim(x12)
        ax[0][3].set_xlim(x03)
        ax[1][3].set_xlim(x13)

        ax[0][0].set_ylim(y00)
        ax[1][0].set_ylim(y10)
        ax[0][1].set_ylim(y01)
        ax[1][1].set_ylim(y11)
        ax[0][2].set_ylim(y02)
        ax[1][2].set_ylim(y12)
        ax[0][3].set_ylim(y03)
        ax[1][3].set_ylim(y13)

    MPHATE_Inset = True
    if MPHATE_Inset:
        left = .34
        bottom = .74
        width = .05
        height = .08
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])

        start = 3
        # Starts off grouping by within-pool position e.g. input
        ax2.scatter(StdTrainET4[13][12*start,0],StdTrainET4[13][12*start,1], c = 'red')
        ax2.scatter(StdTrainET4[13][12 * start + 3, 0], StdTrainET4[13][12 * start + 3, 1], c='red')
        ax2.scatter(StdTrainET4[13][12 * start + 6, 0], StdTrainET4[13][12 * start + 6, 1], c='red')
        ax2.scatter(StdTrainET4[13][12 * start + 9, 0], StdTrainET4[13][12 * start + 9, 1], c='red')
        ax2.scatter(StdTrainET4[13][12 * start + 1, 0], StdTrainET4[13][12 * start + 1, 1], c='blue')
        ax2.scatter(StdTrainET4[13][12 * start + 4, 0], StdTrainET4[13][12 * start + 4, 1], c='blue')
        ax2.scatter(StdTrainET4[13][12 * start + 7, 0], StdTrainET4[13][12 * start + 7, 1], c='blue')
        ax2.scatter(StdTrainET4[13][12 * start + 10, 0], StdTrainET4[13][12 * start + 10, 1], c='blue')
        ax2.scatter(StdTrainET4[13][12 * start + 2, 0], StdTrainET4[13][12 * start + 2, 1], c='green')
        ax2.scatter(StdTrainET4[13][12 * start + 5, 0], StdTrainET4[13][12 * start + 5, 1], c='green')
        ax2.scatter(StdTrainET4[13][12 * start + 8, 0], StdTrainET4[13][12 * start + 8, 1], c='green')
        ax2.scatter(StdTrainET4[13][12 * start + 11, 0], StdTrainET4[13][12 * start + 11, 1], c='green')

        ax[0][1].plot([.007, StdTrainET4[13][12*start+1, 0]],[.034, StdTrainET4[13][12*start+1, 1]], c = 'black')
        ax[0][1].plot([.007, StdTrainET4[13][12*start+2, 0]],[.018, StdTrainET4[13][12*start+2, 1]], c = 'black')
    #plt.show()

    from matplotlib.backends.backend_pdf import PdfPages
    plt.savefig('MPHATE.pdf')

    # JDC wants to add 'thin, semi-transparent' lines connecting trajectories
    # Also wants a new plot showing an inset of the Std Inputs, showing grouping earlier in trajectory...
    # New1) - Connections
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    scprep.plot.scatter2d(StdTrainET4[11], c=time23, ticks=False, label_prefix="M-PHATE", ax=ax)
    ax.plot(StdTrainET4[11][0:-1:12, 0], StdTrainET4[11][0:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][1:-1:12, 0], StdTrainET4[11][1:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][2:-1:12, 0], StdTrainET4[11][2:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][3:-1:12, 0], StdTrainET4[11][3:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][4:-1:12, 0], StdTrainET4[11][4:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][5:-1:12, 0], StdTrainET4[11][5:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][6:-1:12, 0], StdTrainET4[11][6:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][7:-1:12, 0], StdTrainET4[11][7:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][8:-1:12, 0], StdTrainET4[11][8:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][9:-1:12, 0], StdTrainET4[11][9:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][10:-1:12, 0], StdTrainET4[11][10:-1:12, 1], color='blue', alpha=.25)
    ax.plot(StdTrainET4[11][11:-1:12, 0], StdTrainET4[11][11:-1:12, 1], color='blue', alpha=.25)
    plt.show()
    # New2) - Std Inputs Inset
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    scprep.plot.scatter2d(StdTrainET4[13][12*8:12*12,:], c=time23[12*8:12*12], ticks=False, label_prefix="M-PHATE", ax=ax)
    plt.show()



    start = 3
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    scprep.plot.scatter2d(StdTrainET4[13][12 * start:12 * (start + 1), :], c=time23[12 * start:12 * (start + 1)],
                          ticks=False,
                          label_prefix="M-PHATE", ax=ax)
    ax.legend(['t = 3'])
    plt.show()



    start = 15
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    scprep.plot.scatter2d(StdTrainET4[13][12 * start:12 * (start + 1), :], c=time23[12 * start:12 * (start + 1)],
                          ticks=False,
                          label_prefix="M-PHATE", ax=ax)
    ax.legend(['t = 15'])
    plt.show()

    fig, ax = plt.subplots(1,2, figsize = (10,6))
    start = 3
    scprep.plot.scatter2d(StdTrainET4[13][12 * start:12 * (start + 1), :], c=time23[12 * start:12 * (start + 1)],
                          ticks=False,
                          label_prefix="M-PHATE", ax=ax[0])
    ax[0].legend(['t = 3'])
    start = 15
    scprep.plot.scatter2d(StdTrainET4[13][12 * start:12 * (start + 1), :], c=time23[12 * start:12 * (start + 1)],
                          ticks=False,
                          label_prefix="M-PHATE", ax=ax[1])
    ax[1].legend(['t = 15'])
    plt.suptitle('MPHATE Clusters at t=3,15')
    plt.show()

    start = 75
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    scprep.plot.scatter2d(StdTrainET4[13][12 * start:12 * (start + 1), :], c=time23[12 * start:12 * (start + 1)],
                          ticks=False,
                          label_prefix="M-PHATE", ax=ax)
    plt.show()


    ## MPHATE / NTKevec matching
    # Using same as above, e.g. StdTrainET4
    import sklearn.cluster as skc
    from sklearn import metrics

    # Relies on m_phate_data2 and 23, was ids 8 and 4, now 11 and 13
    # Using StdPretrain, StdPretrain3
    MPHATE_task = np.zeros((12, 2, numplot))
    MPHATE_data = np.zeros((12, 2, numplot))
    for i in range(12):
        MPHATE_task[i, :, :] = StdTrainET4[11][i::12, :].T
        MPHATE_data[i, :, :] = StdTrainET4[13][i::12, :].T

    correct_data = np.zeros(numplot)
    correct_task = np.zeros(numplot)
    for t in range(numplot):
        kmeans_data = skc.KMeans(n_clusters=3, random_state=0).fit(MPHATE_data[:, :, t])
        predict_data = kmeans_data.predict(MPHATE_data[:, :, t])
        correct_data[t] = metrics.adjusted_rand_score(predict_data, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
        kmeans_task = skc.KMeans(n_clusters=4, random_state=0).fit(MPHATE_task[:, :, t])
        predict_task = kmeans_task.predict(MPHATE_task[:, :, t])
        correct_task[t] = metrics.adjusted_rand_score(predict_task, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
    fig = plt.figure(figsize=(8, 6))
    plt.plot(correct_data)
    plt.plot(correct_task)
    plt.legend(['Input', 'Task'], fontsize = 14)
    plt.ylabel('Overlap (Adjusted Rand)', fontsize = 15)
    plt.xlabel('Training Progress', fontsize = 15)
    plt.title('Overlap between PNTK and MPHATE predicted groups', fontsize = 15)
    plt.xlim([0,50])
    plt.show()

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    scprep.plot.scatter2d(StdTrainLR4[11], c=time2, ticks=False, label_prefix="M-PHATE", ax=ax[0][0])
    scprep.plot.scatter2d(LargeTrainLR4[11], c=time2, ticks=False, label_prefix="M-PHATE", ax=ax[1][0])
    scprep.plot.scatter2d(StdTrainLR4[13], c=time23, ticks=False, label_prefix="M-PHATE", ax=ax[0][1])
    scprep.plot.scatter2d(LargeTrainLR4[13], c=time23, ticks=False, label_prefix="M-PHATE", ax=ax[1][1])
    scprep.plot.scatter2d(StdTrainLR4[12], c=time22, ticks=False, label_prefix="M-PHATE", ax=ax[0][2])
    scprep.plot.scatter2d(LargeTrainLR4[12], c=time22, ticks=False, label_prefix="M-PHATE", ax=ax[1][2])
    # Optional : output as well
    scprep.plot.scatter2d(StdTrainLR4[14], c=time24, ticks=False, label_prefix="M-PHATE", ax=ax[0][3])
    scprep.plot.scatter2d(LargeTrainLR4[14], c=time24, ticks=False, label_prefix="M-PHATE", ax=ax[1][3])
    # ax[0][0].set_title(f'Test')
    # Text below assumes we are doing the output as well version - else numbers will probably change
    plt.text(-.5, .1, 'Low Init', fontsize=16, rotation=90)
    plt.text(-.5, 0, 'Large Init', fontsize=16, rotation=90)
    plt.text(0, .16, 'Output', fontsize=16, rotation=0)
    plt.text(-.145, .16, 'Task by Inputs', fontsize=16, rotation=0)
    plt.text(-.275, .16, 'Inputs', fontsize=16, rotation=0)
    plt.text(-.41, .16, 'Task', fontsize=16, rotation=0)
    # He also wants text for color bar. Training %?
    plt.text(-.35, .05, 'Training %', fontsize=12, rotation=0)
    plt.text(-.21, .05, 'Training %', fontsize=12, rotation=0)
    plt.text(-.075, .05, 'Training %', fontsize=12, rotation=0)
    plt.text(.065, .05, 'Training %', fontsize=12, rotation=0)
    plt.text(-.35, .152, 'Training %', fontsize=12, rotation=0)
    plt.text(-.21, .152, 'Training %', fontsize=12, rotation=0)
    plt.text(-.075, .152, 'Training %', fontsize=12, rotation=0)
    plt.text(.065, .152, 'Training %', fontsize=12, rotation=0)
    plt.show()

    ## MPHATE / NTKevec matching
    # Using same as above, e.g. StdTrainET4
    import sklearn.cluster as skc
    from sklearn import metrics

    # Relies on m_phate_data2 and 23, was ids 8 and 4, now 11 and 13
    # Using StdPretrain, StdPretrain3
    MPHATE_task = np.zeros((12, 2, numplot))
    MPHATE_data = np.zeros((12, 2, numplot))
    for i in range(12):
        MPHATE_task[i, :, :] = StdTrainLR4[11][i::12, :].T
        MPHATE_data[i, :, :] = StdTrainLR4[13][i::12, :].T

    correct_data = np.zeros(numplot)
    correct_task = np.zeros(numplot)
    for t in range(numplot):
        kmeans_data = skc.KMeans(n_clusters=3, random_state=0).fit(MPHATE_data[:, :, t])
        predict_data = kmeans_data.predict(MPHATE_data[:, :, t])
        correct_data[t] = metrics.adjusted_rand_score(predict_data, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
        kmeans_task = skc.KMeans(n_clusters=4, random_state=0).fit(MPHATE_task[:, :, t])
        predict_task = kmeans_task.predict(MPHATE_task[:, :, t])
        correct_task[t] = metrics.adjusted_rand_score(predict_task, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
    plt.plot(correct_data)
    plt.plot(correct_task)
    plt.legend(['Data', 'Task'])
    plt.ylabel('Overlap (Adjusted Rand)')
    plt.xlabel('Epoch')
    plt.title('Overlap between PNTK and MPHATE predicted groups')
    plt.show()

    # Need 2 more ET (5-6) so we have 3 to compare against for the MPHATE vs NTK Evec analysis
    # Save Rep5, ET
    StdAllTrain11ETRep5 = f'data\Pickle\CohenStdAllTrain11ETRep5.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep5)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep5.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep5 = f'data\Pickle\CohenLargeAllTrain11ETRep5.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep5)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep5.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Save Rep6, ET
    StdAllTrain11ETRep6 = f'data\Pickle\CohenStdAllTrain11ETRep6.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep6)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep6.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep6 = f'data\Pickle\CohenLargeAllTrain11ETRep6.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep6)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep6.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Save Rep7, ET
    StdAllTrain11ETRep7 = f'data\Pickle\CohenStdAllTrain11ETRep7.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep7)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep7.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep7 = f'data\Pickle\CohenLargeAllTrain11ETRep7.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep7)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep7.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Save Rep8, ET
    StdAllTrain11ETRep8 = f'data\Pickle\CohenStdAllTrain11ETRep8.pt'
    torch.save(model.state_dict(), StdAllTrain11ETRep8)
    tostore = [lstore_train, lstore_test, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum, NTKevecs,
               NTKevals, NTK, m_phate_data2, m_phate_data22, m_phate_data23, m_phate_data24]
    with open(f'data\Pickle\CohenStdAllTrain11ETRep8.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    LargeAllTrain11ETRep8 = f'data\Pickle\CohenLargeAllTrain11ETRep8.pt'
    torch.save(model2.state_dict(), LargeAllTrain11ETRep8)
    tostore = [lstore_train2, lstore_test2, taskinds_train, datainds_train, outinds_train, dsum, tsum, osum,
               NTKevecs2,
               NTKevals2, NTK2, m_phate2_data2, m_phate2_data22, m_phate2_data23, m_phate2_data24]
    with open(f'data\Pickle\CohenLargeAllTrain11ETRep8.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(tostore, file)

    # Generate the NTK/MPHATE overlap plots for 4,5, and 6
    # Then plot an aggregate of some sort (mean + std or mean + low alpha examples)


    with open(f'data\Pickle\CohenStdAllTrain11ETRep4.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET4 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep4.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET4 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep5.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET5 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep5.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET5 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep6.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET6 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep6.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET6 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep7.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET7 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep7.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET7 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep8.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET8 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep8.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET8 = pickle.load(file)

    import sklearn.cluster as skc
    from sklearn import metrics

    # Relies on m_phate_data2 and 23, was ids 8 and 4, now 11 and 13
    # Using StdPretrain, StdPretrain3
    MPHATE_task = np.zeros((12, 2, numplot))
    MPHATE_data = np.zeros((12, 2, numplot))
    for i in range(12):
        MPHATE_task[i, :, :] = StdTrainET4[11][i::12, :].T
        MPHATE_data[i, :, :] = StdTrainET4[13][i::12, :].T

    correct_data = np.zeros(numplot)
    correct_task = np.zeros(numplot)
    for t in range(numplot):
        kmeans_data = skc.KMeans(n_clusters=3, random_state=0).fit(MPHATE_data[:, :, t])
        predict_data = kmeans_data.predict(MPHATE_data[:, :, t])
        correct_data[t] = metrics.adjusted_rand_score(predict_data, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
        kmeans_task = skc.KMeans(n_clusters=4, random_state=0).fit(MPHATE_task[:, :, t])
        predict_task = kmeans_task.predict(MPHATE_task[:, :, t])
        correct_task[t] = metrics.adjusted_rand_score(predict_task, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
    fig = plt.figure(figsize=(8, 6))
    plt.plot(correct_data)
    plt.plot(correct_task)
    plt.legend(['Input', 'Task'], fontsize=14)
    plt.ylabel('Overlap (Adjusted Rand)', fontsize=15)
    plt.xlabel('Training Progress', fontsize=15)
    plt.title('Overlap between PNTK and MPHATE predicted groups', fontsize=15)
    plt.xlim([0, 50])
    plt.show()

    MPHATE_task2 = np.zeros((12, 2, numplot))
    MPHATE_data2 = np.zeros((12, 2, numplot))
    for i in range(12):
        MPHATE_task2[i, :, :] = StdTrainET5[11][i::12, :].T
        MPHATE_data2[i, :, :] = StdTrainET5[13][i::12, :].T

    correct_data2 = np.zeros(numplot)
    correct_task2 = np.zeros(numplot)
    for t in range(numplot):
        kmeans_data = skc.KMeans(n_clusters=3, random_state=0).fit(MPHATE_data2[:, :, t])
        predict_data = kmeans_data.predict(MPHATE_data2[:, :, t])
        correct_data2[t] = metrics.adjusted_rand_score(predict_data, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
        kmeans_task = skc.KMeans(n_clusters=4, random_state=0).fit(MPHATE_task2[:, :, t])
        predict_task = kmeans_task.predict(MPHATE_task2[:, :, t])
        correct_task2[t] = metrics.adjusted_rand_score(predict_task, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
    fig = plt.figure(figsize=(8, 6))
    plt.plot(correct_data2)
    plt.plot(correct_task2)
    plt.legend(['Input', 'Task'], fontsize=14)
    plt.ylabel('Overlap (Adjusted Rand)', fontsize=15)
    plt.xlabel('Training Progress', fontsize=15)
    plt.title('Overlap between PNTK and MPHATE predicted groups', fontsize=15)
    plt.xlim([0, 50])
    plt.show()

    MPHATE_task3 = np.zeros((12, 2, numplot))
    MPHATE_data3 = np.zeros((12, 2, numplot))
    for i in range(12):
        MPHATE_task3[i, :, :] = StdTrainET6[11][i::12, :].T
        MPHATE_data3[i, :, :] = StdTrainET6[13][i::12, :].T

    correct_data3 = np.zeros(numplot)
    correct_task3 = np.zeros(numplot)
    for t in range(numplot):
        kmeans_data = skc.KMeans(n_clusters=3, random_state=0).fit(MPHATE_data3[:, :, t])
        predict_data = kmeans_data.predict(MPHATE_data3[:, :, t])
        correct_data3[t] = metrics.adjusted_rand_score(predict_data, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
        kmeans_task = skc.KMeans(n_clusters=4, random_state=0).fit(MPHATE_task3[:, :, t])
        predict_task = kmeans_task.predict(MPHATE_task3[:, :, t])
        correct_task3[t] = metrics.adjusted_rand_score(predict_task, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
    fig = plt.figure(figsize=(8, 6))
    plt.plot(correct_data3)
    plt.plot(correct_task3)
    plt.legend(['Input', 'Task'], fontsize=14)
    plt.ylabel('Overlap (Adjusted Rand)', fontsize=15)
    plt.xlabel('Training Progress', fontsize=15)
    plt.title('Overlap between PNTK and MPHATE predicted groups', fontsize=15)
    plt.xlim([0, 50])
    plt.show()

    MPHATE_task4 = np.zeros((12, 2, numplot))
    MPHATE_data4 = np.zeros((12, 2, numplot))
    for i in range(12):
        MPHATE_task4[i, :, :] = StdTrainET7[11][i::12, :].T
        MPHATE_data4[i, :, :] = StdTrainET7[13][i::12, :].T

    correct_data4 = np.zeros(numplot)
    correct_task4 = np.zeros(numplot)
    for t in range(numplot):
        kmeans_data = skc.KMeans(n_clusters=3, random_state=0).fit(MPHATE_data4[:, :, t])
        predict_data = kmeans_data.predict(MPHATE_data4[:, :, t])
        correct_data4[t] = metrics.adjusted_rand_score(predict_data, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
        kmeans_task = skc.KMeans(n_clusters=4, random_state=0).fit(MPHATE_task4[:, :, t])
        predict_task = kmeans_task.predict(MPHATE_task4[:, :, t])
        correct_task4[t] = metrics.adjusted_rand_score(predict_task, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
    fig = plt.figure(figsize=(8, 6))
    plt.plot(correct_data4)
    plt.plot(correct_task4)
    plt.legend(['Input', 'Task'], fontsize=14)
    plt.ylabel('Overlap (Adjusted Rand)', fontsize=15)
    plt.xlabel('Training Progress', fontsize=15)
    plt.title('Overlap between PNTK and MPHATE predicted groups', fontsize=15)
    plt.xlim([0, 50])
    plt.show()

    MPHATE_task5 = np.zeros((12, 2, numplot))
    MPHATE_data5 = np.zeros((12, 2, numplot))
    for i in range(12):
        MPHATE_task5[i, :, :] = StdTrainET8[11][i::12, :].T
        MPHATE_data5[i, :, :] = StdTrainET8[13][i::12, :].T

    correct_data5 = np.zeros(numplot)
    correct_task5 = np.zeros(numplot)
    for t in range(numplot):
        kmeans_data = skc.KMeans(n_clusters=3, random_state=0).fit(MPHATE_data5[:, :, t])
        predict_data = kmeans_data.predict(MPHATE_data5[:, :, t])
        correct_data5[t] = metrics.adjusted_rand_score(predict_data, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
        kmeans_task = skc.KMeans(n_clusters=4, random_state=0).fit(MPHATE_task5[:, :, t])
        predict_task = kmeans_task.predict(MPHATE_task5[:, :, t])
        correct_task5[t] = metrics.adjusted_rand_score(predict_task, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
    fig = plt.figure(figsize=(8, 6))
    plt.plot(correct_data5)
    plt.plot(correct_task5)
    plt.legend(['Input', 'Task'], fontsize=14)
    plt.ylabel('Overlap (Adjusted Rand)', fontsize=15)
    plt.xlabel('Training Progress', fontsize=15)
    plt.title('Overlap between PNTK and MPHATE predicted groups', fontsize=15)
    plt.xlim([0, 50])
    plt.show()

    # Try mean + low alpha of all 5
    correct_datas = np.vstack((correct_data, correct_data2, correct_data3, correct_data4, correct_data5))
    correct_tasks = np.vstack((correct_task, correct_task2, correct_task3, correct_task4, correct_task5))
    plt.plot(np.mean(correct_datas,0))
    plt.plot(np.mean(correct_tasks,0))
    plt.legend(['Input', 'Task'], fontsize=14)
    plt.plot(correct_datas[:,:].transpose(), color = 'blue', alpha = .25)
    plt.plot(correct_tasks[:, :].transpose(), color= 'orange', alpha=.25)
    plt.ylabel('Overlap (Adjusted Rand)', fontsize=15)
    plt.xlabel('Training Progress', fontsize=15)
    plt.title('Overlap between PNTK and MPHATE predicted groups', fontsize=15)
    plt.xlim([0, 50])
    plt.show()



    # Try mean + sd lines
    correct_datas = np.vstack((correct_data, correct_data2, correct_data3, correct_data4, correct_data5))
    correct_tasks = np.vstack((correct_task, correct_task2, correct_task3, correct_task4, correct_task5))
    plt.errorbar(np.arange(101),np.mean(correct_datas, 0), np.std(correct_datas,0))
    plt.errorbar(np.arange(101),np.mean(correct_tasks, 0), np.std(correct_tasks,0))
    plt.legend(['Input', 'Task'], fontsize=14)
    plt.ylabel('Overlap (Adjusted Rand)', fontsize=15)
    plt.xlabel('Training Progress', fontsize=15)
    plt.title('Overlap between PNTK and MPHATE predicted groups', fontsize=15)
    plt.xlim([0, 50])
    plt.show()

    # Try mean + sd tube
    plt.plot(np.mean(correct_datas, 0))
    plt.plot(np.mean(correct_tasks, 0))
    plt.legend(['Input', 'Task'], fontsize=14)
    plt.ylabel('Overlap (Adjusted Rand)', fontsize=15)
    plt.xlabel('Training Progress', fontsize=15)
    plt.title('Overlap between eNTK and MPHATE predicted groups', fontsize=15)
    plt.xlim([0, 50])
    plt.fill_between(np.arange(101), np.mean(correct_datas,0), np.mean(correct_datas,0) + np.std(correct_datas,0), color = 'blue', alpha = .25)
    plt.fill_between(np.arange(101), np.mean(correct_datas, 0), np.mean(correct_datas, 0) - np.std(correct_datas, 0),
                     color='blue', alpha=.25)
    plt.fill_between(np.arange(101), np.mean(correct_tasks, 0), np.mean(correct_tasks, 0) + np.std(correct_tasks, 0),
                     color='orange', alpha=.25)
    plt.fill_between(np.arange(101), np.mean(correct_tasks, 0), np.mean(correct_tasks, 0) - np.std(correct_tasks, 0),
                     color='orange', alpha=.25)

    plt.show()

    # Looks good!

    # Want to do the same thing for the other plots...not sure if only 3 trials will be enough
    # May need to go to 5
    # Try with what we have, see how it looks

    ### Load all
    # Load Train 1-3
    with open(f'data\Pickle\CohenStdAllTrain11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET3 = pickle.load(file)

    # Load Tuned 1-3
    with open(f'data\Pickle\CohenStdAllTune11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET3 = pickle.load(file)


    ### Generate results
    # Get means - stdmeantrain, largemeantrain, stdmeantune, largemeantune for both ET and LR
    stdmeantrainLR = np.vstack((StdTrainLR1[1], StdTrainLR2[1], StdTrainLR3[1]))
    largemeantrainLR = np.vstack((LargeTrainLR1[1], LargeTrainLR2[1], LargeTrainLR3[1]))
    stdmeantuneLR = np.vstack((StdTuneLR1[1], StdTuneLR2[1], StdTuneLR3[1]))
    largemeantuneLR = np.vstack((LargeTuneLR1[1], LargeTuneLR2[1], LargeTuneLR3[1]))
    stdmeantrainET = np.vstack((StdTrainET1[1], StdTrainET2[1], StdTrainET3[1]))
    largemeantrainET = np.vstack((LargeTrainET1[1], LargeTrainET2[1], LargeTrainET3[1]))
    stdmeantuneET = np.vstack((StdTuneET1[1], StdTuneET2[1], StdTuneET3[1]))
    largemeantuneET = np.vstack((LargeTuneET1[1], LargeTuneET2[1], LargeTuneET3[1]))
    # Plot 1
    figure, axs = plt.subplots(1,2, figsize = (12,6))
    axs[0].semilogy(np.mean(stdmeantrainLR,0))
    axs[0].semilogy(np.mean(largemeantrainLR,0))
    axs[0].legend(['Std', 'Large'], fontsize=14)
    axs[0].fill_between(np.arange(2501), np.mean(stdmeantrainLR, 0),
                        np.mean(stdmeantrainLR, 0) + np.std(stdmeantrainLR, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(2501), np.mean(stdmeantrainLR, 0),
                        np.mean(stdmeantrainLR, 0) - np.std(stdmeantrainLR, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(2501), np.mean(largemeantrainLR, 0),
                        np.mean(largemeantrainLR, 0) + np.std(largemeantrainLR, 0), color
                        ='orange', alpha=.25)
    axs[0].fill_between(np.arange(2501), np.mean(largemeantrainLR, 0),
                        np.mean(largemeantrainLR, 0) - np.std(largemeantrainLR, 0), color
                        ='orange', alpha=.25)
    axs[1].semilogy(np.mean(stdmeantuneLR,0))
    axs[1].semilogy(np.mean(largemeantuneLR,0))
    axs[1].legend(['Std', 'Large'], fontsize=14)
    axs[1].fill_between(np.arange(2501), np.mean(stdmeantuneLR, 0),
                        np.mean(stdmeantuneLR, 0) + np.std(stdmeantuneLR, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(2501), np.mean(stdmeantuneLR, 0),
                        np.mean(stdmeantuneLR, 0) - np.std(stdmeantuneLR, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(2501), np.mean(largemeantuneLR, 0),
                        np.mean(largemeantuneLR, 0) + np.std(largemeantuneLR, 0), color
                        ='orange', alpha=.25)
    axs[1].fill_between(np.arange(2501), np.mean(largemeantuneLR, 0),
                        np.mean(largemeantuneLR, 0) - np.std(largemeantuneLR, 0), color
                        ='orange', alpha=.25)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Generalization Loss', fontsize=16)
    axs[0].set_title('Generalization - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Generalization Loss', fontsize=16)
    axs[1].set_title('Generalization - Multi-Task Tuning', fontsize=16)
    plt.show()
    # Plot 2, same as above but for ET
    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(np.mean(stdmeantrainET, 0))
    axs[0].semilogy(np.mean(largemeantrainET, 0))
    axs[0].legend(['Std', 'Large'], fontsize=14)
    axs[0].fill_between(np.arange(10001), np.mean(stdmeantrainET, 0),
                        np.mean(stdmeantrainET, 0) + np.std(stdmeantrainET, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(10001), np.mean(stdmeantrainET, 0),
                        np.mean(stdmeantrainET, 0) - np.std(stdmeantrainET, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(10001), np.mean(largemeantrainET, 0),
                        np.mean(largemeantrainET, 0) + np.std(largemeantrainET, 0), color
                        ='orange', alpha=.25)
    axs[0].fill_between(np.arange(10001), np.mean(largemeantrainET, 0),
                        np.mean(largemeantrainET, 0) - np.std(largemeantrainET, 0), color
                        ='orange', alpha=.25)
    axs[1].semilogy(np.mean(stdmeantuneET, 0))
    axs[1].semilogy(np.mean(largemeantuneET, 0))
    axs[1].legend(['Std', 'Large'], fontsize=14)
    axs[1].fill_between(np.arange(10001), np.mean(stdmeantuneET, 0),
                        np.mean(stdmeantuneET, 0) + np.std(stdmeantuneET, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(10001), np.mean(stdmeantuneET, 0),
                        np.mean(stdmeantuneET, 0) - np.std(stdmeantuneET, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(10001), np.mean(largemeantuneET, 0),
                        np.mean(largemeantuneET, 0) + np.std(largemeantuneET, 0), color
                        ='orange', alpha=.25)
    axs[1].fill_between(np.arange(10001), np.mean(largemeantuneET, 0),
                        np.mean(largemeantuneET, 0) - np.std(largemeantuneET, 0), color
                        ='orange', alpha=.25)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Generalization Loss', fontsize=16)
    axs[0].set_title('Generalization - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Generalization Loss', fontsize=16)
    axs[1].set_title('Generalization - Multi-Task Tuning', fontsize=16)
    plt.show()

    # Have another version that also includes train loss? Swap to mean from single example...
    stdmeantrainLR_train = np.vstack((StdTrainLR1[0], StdTrainLR2[0], StdTrainLR3[0]))
    largemeantrainLR_train = np.vstack((LargeTrainLR1[0], LargeTrainLR2[0], LargeTrainLR3[0]))
    stdmeantuneLR_train = np.vstack((StdTuneLR1[0], StdTuneLR2[0], StdTuneLR3[0]))
    largemeantuneLR_train = np.vstack((LargeTuneLR1[0], LargeTuneLR2[0], LargeTuneLR3[0]))
    stdmeantrainET_train = np.vstack((StdTrainET1[0], StdTrainET2[0], StdTrainET3[0]))
    largemeantrainET_train = np.vstack((LargeTrainET1[0], LargeTrainET2[0], LargeTrainET3[0]))
    stdmeantuneET_train = np.vstack((StdTuneET1[0], StdTuneET2[0], StdTuneET3[0]))
    largemeantuneET_train = np.vstack((LargeTuneET1[0], LargeTuneET2[0], LargeTuneET3[0]))

    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(np.mean(stdmeantrainLR,0))
    axs[0].semilogy(np.mean(largemeantrainLR,0))
    axs[0].semilogy(np.mean(stdmeantrainLR_train,0), c = '#1f77b4', linestyle = 'dotted')
    axs[0].semilogy(np.mean(largemeantrainLR_train,0), c = '#ff7f0e', linestyle = 'dotted')
    axs[0].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[1].semilogy(np.mean(stdmeantuneLR,0))
    axs[1].semilogy(np.mean(largemeantuneLR,0))
    axs[1].semilogy(np.mean(stdmeantuneLR_train,0), c = '#1f77b4', linestyle = 'dotted')
    axs[1].semilogy(np.mean(largemeantuneLR_train,0), c = '#ff7f0e', linestyle = 'dotted')
    axs[1].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Loss', fontsize=16)
    axs[0].set_title('Losses - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Loss', fontsize=16)
    axs[1].set_title('Losses - Multi-Task Tuning', fontsize=16)
    plt.show()

    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(np.mean(stdmeantrainET,0))
    axs[0].semilogy(np.mean(largemeantrainET,0))
    axs[0].semilogy(np.mean(stdmeantrainET_train,0), c = '#1f77b4', linestyle = 'dotted')
    axs[0].semilogy(np.mean(largemeantrainET_train,0), c = '#ff7f0e', linestyle = 'dotted')
    axs[0].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[1].semilogy(np.mean(stdmeantuneET,0))
    axs[1].semilogy(np.mean(largemeantuneET,0))
    axs[1].semilogy(np.mean(stdmeantuneET_train,0), c = '#1f77b4', linestyle = 'dotted')
    axs[1].semilogy(np.mean(largemeantuneET_train,0), c = '#ff7f0e', linestyle = 'dotted')
    axs[1].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Loss', fontsize=16)
    axs[0].set_title('Losses - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Loss', fontsize=16)
    axs[1].set_title('Losses - Multi-Task Tuning', fontsize=16)
    plt.show()

    ##### Upgrade from 3 to 10. Don't save MPHATE data.
    ### Load all
    # Load Train 1-3
    with open(f'data\Pickle\CohenStdAllTrain11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET3 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep14.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR4 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep14.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR4 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep14.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET4 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep14.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET4 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep15.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR5 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep15.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR5 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep15.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET5 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep15.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET5 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep16.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR6 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep16.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR6 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep16.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET6 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep16.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET6 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep17.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR7 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep17.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR7 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep17.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET7 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep17.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET7 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep18.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR8 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep18.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR8 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep18.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET8 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep18.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET8 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep19.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR9 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep19.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR9 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep19.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET9 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep19.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET9 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11LRRep20.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainLR10 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11LRRep20.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainLR10 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTrain11ETRep20.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTrainET10 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTrain11ETRep20.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTrainET10 = pickle.load(file)


    # Load Tuned 1-3
    with open(f'data\Pickle\CohenStdAllTune11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET1 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep1.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET1 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET2 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep2.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET2 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR3 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET3 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep3.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET3 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep14.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR4 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep14.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR4 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep14.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET4 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep14.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET4 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep15.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR5 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep15.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR5 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep15.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET5 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep15.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET5 = pickle.load(file)


    with open(f'data\Pickle\CohenStdAllTune11LRRep16.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR6 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep16.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR6 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep16.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET6 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep16.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET6 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep17.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR7 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep17.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR7 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep17.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET7 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep17.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET7 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep18.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR8 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep18.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR8 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep18.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET8 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep18.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET8 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep19.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR9 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep19.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR9 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep19.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET9 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep19.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET9 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11LRRep20.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneLR10 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11LRRep20.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneLR10 = pickle.load(file)

    with open(f'data\Pickle\CohenStdAllTune11ETRep20.pkl', 'rb') as file:
        # Call load method to deserialze
        StdTuneET10 = pickle.load(file)

    with open(f'data\Pickle\CohenLargeAllTune11ETRep20.pkl', 'rb') as file:
        # Call load method to deserialze
        LargeTuneET10 = pickle.load(file)

    ### Generate results for 10 trial version
    # Get means - stdmeantrain, largemeantrain, stdmeantune, largemeantune for both ET and LR
    stdmeantrainLR = np.vstack((StdTrainLR1[1], StdTrainLR2[1], StdTrainLR3[1], StdTrainLR4[1], StdTrainLR5[1], StdTrainLR6[1], StdTrainLR7[1], StdTrainLR8[1], StdTrainLR9[1], StdTrainLR10[1]))
    largemeantrainLR = np.vstack((LargeTrainLR1[1], LargeTrainLR2[1], LargeTrainLR3[1], LargeTrainLR4[1], LargeTrainLR5[1], LargeTrainLR6[1], LargeTrainLR7[1], LargeTrainLR8[1], LargeTrainLR9[1], LargeTrainLR10[1]))
    stdmeantuneLR = np.vstack((StdTuneLR1[1], StdTuneLR2[1], StdTuneLR3[1], StdTuneLR4[1], StdTuneLR5[1], StdTuneLR6[1], StdTuneLR7[1], StdTuneLR8[1], StdTuneLR9[1], StdTuneLR10[1]))
    largemeantuneLR = np.vstack((LargeTuneLR1[1], LargeTuneLR2[1], LargeTuneLR3[1], LargeTuneLR4[1], LargeTuneLR5[1], LargeTuneLR6[1], LargeTuneLR7[1], LargeTuneLR8[1], LargeTuneLR9[1], LargeTuneLR10[1]))
    stdmeantrainET = np.vstack((StdTrainET1[1], StdTrainET2[1], StdTrainET3[1], StdTrainET4[1], StdTrainET5[1], StdTrainET6[1], StdTrainET7[1], StdTrainET8[1], StdTrainET9[1], StdTrainET10[1]))
    largemeantrainET = np.vstack((LargeTrainET1[1], LargeTrainET2[1], LargeTrainET3[1], LargeTrainET4[1], LargeTrainET5[1], LargeTrainET6[1], LargeTrainET7[1], LargeTrainET8[1], LargeTrainET9[1], LargeTrainET10[1]))
    stdmeantuneET = np.vstack((StdTuneET1[1], StdTuneET2[1], StdTuneET3[1], StdTuneET4[1], StdTuneET5[1], StdTuneET6[1], StdTuneET7[1], StdTuneET8[1], StdTuneET9[1], StdTuneET10[1]))
    largemeantuneET = np.vstack((LargeTuneET1[1], LargeTuneET2[1], LargeTuneET3[1], LargeTuneET4[1], LargeTuneET5[1], LargeTuneET6[1], LargeTuneET7[1], LargeTuneET8[1], LargeTuneET9[1], LargeTuneET10[1]))
    # Plot 1
    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(np.mean(stdmeantrainLR, 0))
    axs[0].semilogy(np.mean(largemeantrainLR, 0))
    axs[0].legend(['Std', 'Large'], fontsize=14)
    axs[0].fill_between(np.arange(2501), np.mean(stdmeantrainLR, 0),
                        np.mean(stdmeantrainLR, 0) + np.std(stdmeantrainLR, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(2501), np.mean(stdmeantrainLR, 0),
                        np.mean(stdmeantrainLR, 0) - np.std(stdmeantrainLR, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(2501), np.mean(largemeantrainLR, 0),
                        np.mean(largemeantrainLR, 0) + np.std(largemeantrainLR, 0), color
                        ='orange', alpha=.25)
    axs[0].fill_between(np.arange(2501), np.mean(largemeantrainLR, 0),
                        np.mean(largemeantrainLR, 0) - np.std(largemeantrainLR, 0), color
                        ='orange', alpha=.25)
    axs[1].semilogy(np.mean(stdmeantuneLR, 0))
    axs[1].semilogy(np.mean(largemeantuneLR, 0))
    axs[1].legend(['Std', 'Large'], fontsize=14)
    axs[1].fill_between(np.arange(2501), np.mean(stdmeantuneLR, 0),
                        np.mean(stdmeantuneLR, 0) + np.std(stdmeantuneLR, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(2501), np.mean(stdmeantuneLR, 0),
                        np.mean(stdmeantuneLR, 0) - np.std(stdmeantuneLR, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(2501), np.mean(largemeantuneLR, 0),
                        np.mean(largemeantuneLR, 0) + np.std(largemeantuneLR, 0), color
                        ='orange', alpha=.25)
    axs[1].fill_between(np.arange(2501), np.mean(largemeantuneLR, 0),
                        np.mean(largemeantuneLR, 0) - np.std(largemeantuneLR, 0), color
                        ='orange', alpha=.25)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Generalization Loss', fontsize=16)
    axs[0].set_title('Generalization - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Generalization Loss', fontsize=16)
    axs[1].set_title('Generalization - Multi-Task Tuning', fontsize=16)
    plt.show()
    # Plot 2, same as above but for ET
    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(np.mean(stdmeantrainET, 0))
    axs[0].semilogy(np.mean(largemeantrainET, 0))
    axs[0].legend(['Std', 'Large'], fontsize=14)
    axs[0].fill_between(np.arange(10001), np.mean(stdmeantrainET, 0),
                        np.mean(stdmeantrainET, 0) + np.std(stdmeantrainET, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(10001), np.mean(stdmeantrainET, 0),
                        np.mean(stdmeantrainET, 0) - np.std(stdmeantrainET, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(10001), np.mean(largemeantrainET, 0),
                        np.mean(largemeantrainET, 0) + np.std(largemeantrainET, 0), color
                        ='orange', alpha=.25)
    axs[0].fill_between(np.arange(10001), np.mean(largemeantrainET, 0),
                        np.mean(largemeantrainET, 0) - np.std(largemeantrainET, 0), color
                        ='orange', alpha=.25)
    axs[1].semilogy(np.mean(stdmeantuneET, 0))
    axs[1].semilogy(np.mean(largemeantuneET, 0))
    axs[1].legend(['Std', 'Large'], fontsize=14)
    axs[1].fill_between(np.arange(10001), np.mean(stdmeantuneET, 0),
                        np.mean(stdmeantuneET, 0) + np.std(stdmeantuneET, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(10001), np.mean(stdmeantuneET, 0),
                        np.mean(stdmeantuneET, 0) - np.std(stdmeantuneET, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(10001), np.mean(largemeantuneET, 0),
                        np.mean(largemeantuneET, 0) + np.std(largemeantuneET, 0), color
                        ='orange', alpha=.25)
    axs[1].fill_between(np.arange(10001), np.mean(largemeantuneET, 0),
                        np.mean(largemeantuneET, 0) - np.std(largemeantuneET, 0), color
                        ='orange', alpha=.25)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Generalization Loss', fontsize=16)
    axs[0].set_title('Generalization - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Generalization Loss', fontsize=16)
    axs[1].set_title('Generalization - Multi-Task Tuning', fontsize=16)
    plt.show()

    # Have another version that also includes train loss? Swap to mean from single example...
    stdmeantrainLR_train = np.vstack((StdTrainLR1[0], StdTrainLR2[0], StdTrainLR3[0], StdTrainLR4[0], StdTrainLR5[0], StdTrainLR6[0], StdTrainLR7[0], StdTrainLR8[0], StdTrainLR9[0], StdTrainLR10[0]))
    largemeantrainLR_train = np.vstack((LargeTrainLR1[0], LargeTrainLR2[0], LargeTrainLR3[0], LargeTrainLR4[0], LargeTrainLR5[0], LargeTrainLR6[0], LargeTrainLR7[0], LargeTrainLR8[0], LargeTrainLR9[0], LargeTrainLR10[0]))
    stdmeantuneLR_train = np.vstack((StdTuneLR1[0], StdTuneLR2[0], StdTuneLR3[0], StdTuneLR4[0], StdTuneLR5[0], StdTuneLR6[0], StdTuneLR7[0], StdTuneLR8[0], StdTuneLR9[0], StdTuneLR10[0]))
    largemeantuneLR_train = np.vstack((LargeTuneLR1[0], LargeTuneLR2[0], LargeTuneLR3[0], LargeTuneLR4[0], LargeTuneLR5[0], LargeTuneLR6[0], LargeTuneLR7[0], LargeTuneLR8[0], LargeTuneLR9[0], LargeTuneLR10[0]))
    stdmeantrainET_train = np.vstack((StdTrainET1[0], StdTrainET2[0], StdTrainET3[0], StdTrainET4[0], StdTrainET5[0], StdTrainET6[0], StdTrainET7[0], StdTrainET8[0], StdTrainET9[0], StdTrainET10[0]))
    largemeantrainET_train = np.vstack((LargeTrainET1[0], LargeTrainET2[0], LargeTrainET3[0], LargeTrainET4[0], LargeTrainET5[0], LargeTrainET6[0], LargeTrainET7[0], LargeTrainET8[0], LargeTrainET9[0], LargeTrainET10[0]))
    stdmeantuneET_train = np.vstack((StdTuneET1[0], StdTuneET2[0], StdTuneET3[0], StdTuneET4[0], StdTuneET5[0], StdTuneET6[0], StdTuneET7[0], StdTuneET8[0], StdTuneET9[0], StdTuneET10[0]))
    largemeantuneET_train = np.vstack((LargeTuneET1[0], LargeTuneET2[0], LargeTuneET3[0], LargeTuneET4[0], LargeTuneET5[0], LargeTuneET6[0], LargeTuneET7[0], LargeTuneET8[0], LargeTuneET9[0], LargeTuneET10[0]))

    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(np.mean(stdmeantrainLR, 0))
    axs[0].semilogy(np.mean(largemeantrainLR, 0))
    axs[0].semilogy(np.mean(stdmeantrainLR_train, 0), c='#1f77b4', linestyle='dotted')
    axs[0].semilogy(np.mean(largemeantrainLR_train, 0), c='#ff7f0e', linestyle='dotted')
    axs[0].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[1].semilogy(np.mean(stdmeantuneLR, 0))
    axs[1].semilogy(np.mean(largemeantuneLR, 0))
    axs[1].semilogy(np.mean(stdmeantuneLR_train, 0), c='#1f77b4', linestyle='dotted')
    axs[1].semilogy(np.mean(largemeantuneLR_train, 0), c='#ff7f0e', linestyle='dotted')
    axs[1].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Loss', fontsize=16)
    axs[0].set_title('Losses - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Loss', fontsize=16)
    axs[1].set_title('Losses - Multi-Task Tuning', fontsize=16)
    plt.show()

    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(np.mean(stdmeantrainET, 0))
    axs[0].semilogy(np.mean(largemeantrainET, 0))
    axs[0].semilogy(np.mean(stdmeantrainET_train, 0), c='#1f77b4', linestyle='dotted')
    axs[0].semilogy(np.mean(largemeantrainET_train, 0), c='#ff7f0e', linestyle='dotted')
    axs[0].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[1].semilogy(np.mean(stdmeantuneET, 0))
    axs[1].semilogy(np.mean(largemeantuneET, 0))
    axs[1].semilogy(np.mean(stdmeantuneET_train, 0), c='#1f77b4', linestyle='dotted')
    axs[1].semilogy(np.mean(largemeantuneET_train, 0), c='#ff7f0e', linestyle='dotted')
    axs[1].legend(['Std - Test', 'Large - Test', 'Std - Train', 'Large - Train'], fontsize=14)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Loss', fontsize=16)
    axs[0].set_title('Losses - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Loss', fontsize=16)
    axs[1].set_title('Losses - Multi-Task Tuning', fontsize=16)
    plt.show()
    # Looks terrible

    plt.semilogy(StdTrainET1[1])
    plt.semilogy(LargeTrainET1[1])
    plt.show()

    plt.semilogy(StdTrainET2[1])
    plt.semilogy(LargeTrainET2[1])
    plt.show()

    plt.semilogy(StdTrainET3[1])
    plt.semilogy(LargeTrainET3[1])
    plt.show()

    plt.semilogy(StdTrainET4[1])
    plt.semilogy(LargeTrainET4[1])
    plt.show()

    plt.semilogy(StdTrainET5[1])
    plt.semilogy(LargeTrainET5[1])
    plt.show()

    plt.semilogy(StdTrainET6[1])
    plt.semilogy(LargeTrainET6[1])
    plt.show()

    plt.semilogy(StdTrainET7[1])
    plt.semilogy(LargeTrainET7[1])
    plt.show()

    plt.semilogy(StdTrainET8[1])
    plt.semilogy(LargeTrainET8[1])
    plt.show()

    plt.semilogy(StdTrainET9[1])
    plt.semilogy(LargeTrainET9[1])
    plt.show()

    plt.semilogy(StdTrainET10[1])
    plt.semilogy(LargeTrainET10[1])
    plt.show()
        # But every single one std > large by the end
    # Problem is that std, large are linked per trial, and then doing sd over log scale?
    # E.g. the valid statistic would be p-val of whether (large-std) > 0?
    # Try two more plots: large-std mean/sd, with 0 line marked?
    # Semilogy wouldn't work as it is sometimes negative. Std would have a huge issue with scale...\
    # Just try one for starters

    trainmeandeltaLR = np.vstack((LargeTrainLR1[1]-StdTrainLR1[1], LargeTrainLR2[1]-StdTrainLR2[1], LargeTrainLR3[1]-StdTrainLR3[1], LargeTrainLR4[1]-StdTrainLR4[1], LargeTrainLR5[1]-StdTrainLR5[1], LargeTrainLR6[1]-StdTrainLR6[1], LargeTrainLR7[1]-StdTrainLR7[1], LargeTrainLR8[1]-StdTrainLR8[1], LargeTrainLR9[1]-StdTrainLR9[1], LargeTrainLR10[1]-StdTrainLR10[1]))
    trainmeandeltaET = np.vstack((LargeTrainET1[1]-StdTrainET1[1], LargeTrainET2[1]-StdTrainET2[1], LargeTrainET3[1]-StdTrainET3[1], LargeTrainET4[1]-StdTrainET4[1], LargeTrainET5[1]-StdTrainET5[1], LargeTrainET6[1]-StdTrainET6[1], LargeTrainET7[1]-StdTrainET7[1], LargeTrainET8[1]-StdTrainET8[1], LargeTrainET9[1]-StdTrainET9[1], LargeTrainET10[1]-StdTrainET10[1]))
    tunemeandeltaLR = np.vstack((LargeTuneLR1[1]-StdTuneLR1[1], LargeTuneLR2[1]-StdTuneLR2[1], LargeTuneLR3[1]-StdTuneLR3[1], LargeTuneLR4[1]-StdTuneLR4[1], LargeTuneLR5[1]-StdTuneLR5[1], LargeTuneLR6[1]-StdTuneLR6[1], LargeTuneLR7[1]-StdTuneLR7[1], LargeTuneLR8[1]-StdTuneLR8[1], LargeTuneLR9[1]-StdTuneLR9[1], LargeTuneLR10[1]-StdTuneLR10[1]))
    tunemeandeltaET = np.vstack((LargeTuneET1[1]-StdTuneET1[1], LargeTuneET2[1]-StdTuneET2[1], LargeTuneET3[1]-StdTuneET3[1], LargeTuneET4[1]-StdTuneET4[1], LargeTuneET5[1]-StdTuneET5[1], LargeTuneET6[1]-StdTuneET6[1], LargeTuneET7[1]-StdTuneET7[1], LargeTuneET8[1]-StdTuneET8[1], LargeTuneET9[1]-StdTuneET9[1], LargeTuneET10[1]-StdTuneET10[1]))

    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].semilogy(np.mean(trainmeandeltaLR, 0))
    axs[0].legend(['Delta'], fontsize=14)
    axs[0].fill_between(np.arange(2501), np.mean(trainmeandeltaLR, 0),
                        np.mean(trainmeandeltaLR, 0) + np.std(trainmeandeltaLR, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(2501), np.mean(trainmeandeltaLR, 0),
                        np.mean(trainmeandeltaLR, 0) - np.std(trainmeandeltaLR, 0), color
                        ='blue', alpha=.25)
    axs[1].semilogy(np.mean(trainmeandeltaET, 0))
    axs[1].legend(['Delta'], fontsize=14)
    axs[1].fill_between(np.arange(10001), np.mean(trainmeandeltaET, 0),
                        np.mean(trainmeandeltaET, 0) + np.std(trainmeandeltaET, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(10001), np.mean(trainmeandeltaET, 0),
                        np.mean(trainmeandeltaET, 0) - np.std(trainmeandeltaET, 0), color
                        ='blue', alpha=.25)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Generalization Loss', fontsize=16)
    axs[0].set_title('Generalization - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Generalization Loss', fontsize=16)
    axs[1].set_title('Generalization - Multi-Task Tuning', fontsize=16)
    plt.show()

    # Try regular scale version. Put a dotted red line through 0

    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(np.mean(trainmeandeltaLR, 0))
    axs[0].hlines(0, 0, 2501, color='red', linestyle='dashed')
    axs[0].legend(['Large Loss - Std Loss'], fontsize=14)
    axs[0].fill_between(np.arange(2501), np.mean(trainmeandeltaLR, 0),
                        np.mean(trainmeandeltaLR, 0) + np.std(trainmeandeltaLR, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(2501), np.mean(trainmeandeltaLR, 0),
                        np.mean(trainmeandeltaLR, 0) - np.std(trainmeandeltaLR, 0), color
                        ='blue', alpha=.25)
    axs[1].plot(np.mean(tunemeandeltaLR, 0))
    axs[1].hlines(0, 0, 2501, color= 'red', linestyle = 'dashed')
    axs[1].legend(['Large Loss - Std Loss'], fontsize=14)
    axs[1].fill_between(np.arange(2501), np.mean(tunemeandeltaLR, 0),
                        np.mean(tunemeandeltaLR, 0) + np.std(tunemeandeltaLR, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(2501), np.mean(tunemeandeltaLR, 0),
                        np.mean(tunemeandeltaLR, 0) - np.std(tunemeandeltaLR, 0), color
                        ='blue', alpha=.25)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Delta Generalization Loss', fontsize=16)
    axs[0].set_title('Delta Generalization - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Delta Generalization Loss', fontsize=16)
    axs[1].set_title('Delta Generalization - Multi-Task Tuning', fontsize=16)
    axs[0].set_ylim(-5,15)
    axs[1].set_ylim(-40,10)
    plt.show()

    figure, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(np.mean(trainmeandeltaET, 0))
    axs[0].hlines(0, 0, 10001, color='red', linestyle='dashed')
    axs[0].legend(['Large Loss - Std Loss'], fontsize=14)
    axs[0].fill_between(np.arange(10001), np.mean(trainmeandeltaET, 0),
                        np.mean(trainmeandeltaET, 0) + np.std(trainmeandeltaET, 0), color
                        ='blue', alpha=.25)
    axs[0].fill_between(np.arange(10001), np.mean(trainmeandeltaET, 0),
                        np.mean(trainmeandeltaET, 0) - np.std(trainmeandeltaET, 0), color
                        ='blue', alpha=.25)
    axs[1].plot(np.mean(tunemeandeltaET, 0))
    axs[1].hlines(0, 0, 10001, color='red', linestyle='dashed')
    axs[1].legend(['Large Loss - Std Loss'], fontsize=14)
    axs[1].fill_between(np.arange(10001), np.mean(tunemeandeltaET, 0),
                        np.mean(tunemeandeltaET, 0) + np.std(tunemeandeltaET, 0), color
                        ='blue', alpha=.25)
    axs[1].fill_between(np.arange(10001), np.mean(tunemeandeltaET, 0),
                        np.mean(tunemeandeltaET, 0) - np.std(tunemeandeltaET, 0), color
                        ='blue', alpha=.25)
    axs[0].set_xlabel('Iteration', fontsize=16)
    axs[0].set_ylabel('Delta Generalization Loss', fontsize=16)
    axs[0].set_title('Delta Generalization - Single Task Training', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=16)
    axs[1].set_ylabel('Delta Generalization Loss', fontsize=16)
    axs[1].set_title('Delta Generalization - Multi-Task Tuning', fontsize=16)
    axs[0].set_ylim(-5, 10)
    axs[1].set_ylim(-40, 10)
    plt.show()

    ### Correlations based on the 10 trials:

    # Analysis 1 - Training

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    silhouette_scores = np.zeros(40)
    datuse = np.zeros((40, 500, 9))
    datuse[0, :, :] = np.transpose(StdTrainET1[8][:, :, 0])
    datuse[1, :, :] = np.transpose(StdTrainET2[8][:, :, 0])
    datuse[2, :, :] = np.transpose(StdTrainET3[8][:, :, 0])
    datuse[3, :, :] = np.transpose(StdTrainET4[8][:, :, 0])
    datuse[4, :, :] = np.transpose(StdTrainET5[8][:, :, 0])
    datuse[5, :, :] = np.transpose(StdTrainET6[8][:, :, 0])
    datuse[6, :, :] = np.transpose(StdTrainET7[8][:, :, 0])
    datuse[7, :, :] = np.transpose(StdTrainET8[8][:, :, 0])
    datuse[8, :, :] = np.transpose(StdTrainET9[8][:, :, 0])
    datuse[9, :, :] = np.transpose(StdTrainET10[8][:, :, 0])

    datuse[10, :, :] = np.transpose(StdTrainLR1[8][:, :, 0])
    datuse[11, :, :] = np.transpose(StdTrainLR2[8][:, :, 0])
    datuse[12, :, :] = np.transpose(StdTrainLR3[8][:, :, 0])
    datuse[13, :, :] = np.transpose(StdTrainLR4[8][:, :, 0])
    datuse[14, :, :] = np.transpose(StdTrainLR5[8][:, :, 0])
    datuse[15, :, :] = np.transpose(StdTrainLR6[8][:, :, 0])
    datuse[16, :, :] = np.transpose(StdTrainLR7[8][:, :, 0])
    datuse[17, :, :] = np.transpose(StdTrainLR8[8][:, :, 0])
    datuse[18, :, :] = np.transpose(StdTrainLR9[8][:, :, 0])
    datuse[19, :, :] = np.transpose(StdTrainLR10[8][:, :, 0])

    datuse[20, :, :] = np.transpose(LargeTrainET1[8][:, :, 0])
    datuse[21, :, :] = np.transpose(LargeTrainET2[8][:, :, 0])
    datuse[22, :, :] = np.transpose(LargeTrainET3[8][:, :, 0])
    datuse[23, :, :] = np.transpose(LargeTrainET4[8][:, :, 0])
    datuse[24, :, :] = np.transpose(LargeTrainET5[8][:, :, 0])
    datuse[25, :, :] = np.transpose(LargeTrainET6[8][:, :, 0])
    datuse[26, :, :] = np.transpose(LargeTrainET7[8][:, :, 0])
    datuse[27, :, :] = np.transpose(LargeTrainET8[8][:, :, 0])
    datuse[28, :, :] = np.transpose(LargeTrainET9[8][:, :, 0])
    datuse[29, :, :] = np.transpose(LargeTrainET10[8][:, :, 0])

    datuse[30, :, :] = np.transpose(LargeTrainLR1[8][:, :, 0])
    datuse[31, :, :] = np.transpose(LargeTrainLR2[8][:, :, 0])
    datuse[32, :, :] = np.transpose(LargeTrainLR3[8][:, :, 0])
    datuse[33, :, :] = np.transpose(LargeTrainLR4[8][:, :, 0])
    datuse[34, :, :] = np.transpose(LargeTrainLR5[8][:, :, 0])
    datuse[35, :, :] = np.transpose(LargeTrainLR6[8][:, :, 0])
    datuse[36, :, :] = np.transpose(LargeTrainLR7[8][:, :, 0])
    datuse[37, :, :] = np.transpose(LargeTrainLR8[8][:, :, 0])
    datuse[38, :, :] = np.transpose(LargeTrainLR9[8][:, :, 0])
    datuse[39, :, :] = np.transpose(LargeTrainLR10[8][:, :, 0])

    for i in range(40):
        n_clusters = 12  # 4x3
        X = datuse[i, :, :]
        clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores[i] = silhouette_avg

    finaltestlosses_train = np.zeros(40)
    finaltestlosses_train[0] = StdTrainET1[1][-1]
    finaltestlosses_train[1] = StdTrainET2[1][-1]
    finaltestlosses_train[2] = StdTrainET3[1][-1]
    finaltestlosses_train[3] = StdTrainET4[1][-1]
    finaltestlosses_train[4] = StdTrainET5[1][-1]
    finaltestlosses_train[5] = StdTrainET6[1][-1]
    finaltestlosses_train[6] = StdTrainET7[1][-1]
    finaltestlosses_train[7] = StdTrainET8[1][-1]
    finaltestlosses_train[8] = StdTrainET9[1][-1]
    finaltestlosses_train[9] = StdTrainET10[1][-1]

    finaltestlosses_train[10] = StdTrainLR1[1][-1]
    finaltestlosses_train[11] = StdTrainLR2[1][-1]
    finaltestlosses_train[12] = StdTrainLR3[1][-1]
    finaltestlosses_train[13] = StdTrainLR4[1][-1]
    finaltestlosses_train[14] = StdTrainLR5[1][-1]
    finaltestlosses_train[15] = StdTrainLR6[1][-1]
    finaltestlosses_train[16] = StdTrainLR7[1][-1]
    finaltestlosses_train[17] = StdTrainLR8[1][-1]
    finaltestlosses_train[18] = StdTrainLR9[1][-1]
    finaltestlosses_train[19] = StdTrainLR10[1][-1]

    finaltestlosses_train[20] = LargeTrainET1[1][-1]
    finaltestlosses_train[21] = LargeTrainET2[1][-1]
    finaltestlosses_train[22] = LargeTrainET3[1][-1]
    finaltestlosses_train[23] = LargeTrainET4[1][-1]
    finaltestlosses_train[24] = LargeTrainET5[1][-1]
    finaltestlosses_train[25] = LargeTrainET6[1][-1]
    finaltestlosses_train[26] = LargeTrainET7[1][-1]
    finaltestlosses_train[27] = LargeTrainET8[1][-1]
    finaltestlosses_train[28] = LargeTrainET9[1][-1]
    finaltestlosses_train[29] = LargeTrainET10[1][-1]

    finaltestlosses_train[30] = LargeTrainLR1[1][-1]
    finaltestlosses_train[31] = LargeTrainLR2[1][-1]
    finaltestlosses_train[32] = LargeTrainLR3[1][-1]
    finaltestlosses_train[33] = LargeTrainLR4[1][-1]
    finaltestlosses_train[34] = LargeTrainLR5[1][-1]
    finaltestlosses_train[35] = LargeTrainLR6[1][-1]
    finaltestlosses_train[36] = LargeTrainLR7[1][-1]
    finaltestlosses_train[37] = LargeTrainLR8[1][-1]
    finaltestlosses_train[38] = LargeTrainLR9[1][-1]
    finaltestlosses_train[39] = LargeTrainLR10[1][-1]

    corr1 = np.corrcoef(silhouette_scores, finaltestlosses_train)[0, 1]

    silhouette_scores_1 = np.zeros(20)
    silhouette_scores_1[:10] = silhouette_scores[:10]
    silhouette_scores_1[10:20] = silhouette_scores[20:30]
    silhouette_scores_2 = np.zeros(20)
    silhouette_scores_2[:10] = silhouette_scores[10:20]
    silhouette_scores_2[10:20] = silhouette_scores[30:40]
    finaltestlosses_train_1 = np.zeros(20)
    finaltestlosses_train_1[:10] = finaltestlosses_train[:10]
    finaltestlosses_train_1[10:20] = finaltestlosses_train[20:30]
    finaltestlosses_train_2 = np.zeros(20)
    finaltestlosses_train_2[:10] = finaltestlosses_train[10:20]
    finaltestlosses_train_2[10:20] = finaltestlosses_train[30:40]
    corr1_1 = np.corrcoef(silhouette_scores_1, finaltestlosses_train_1)[0, 1]
    corr1_2 = np.corrcoef(silhouette_scores_2, finaltestlosses_train_2)[0, 1]





    # Analysis 2 - Tuning

    finaltestlosses_tune = np.zeros(40)
    finaltestlosses_tune[0] = StdTuneET1[1][-1]
    finaltestlosses_tune[1] = StdTuneET2[1][-1]
    finaltestlosses_tune[2] = StdTuneET3[1][-1]
    finaltestlosses_tune[3] = StdTuneET4[1][-1]
    finaltestlosses_tune[4] = StdTuneET5[1][-1]
    finaltestlosses_tune[5] = StdTuneET6[1][-1]
    finaltestlosses_tune[6] = StdTuneET7[1][-1]
    finaltestlosses_tune[7] = StdTuneET8[1][-1]
    finaltestlosses_tune[8] = StdTuneET9[1][-1]
    finaltestlosses_tune[9] = StdTuneET10[1][-1]

    finaltestlosses_tune[10] = StdTuneLR1[1][-1]
    finaltestlosses_tune[11] = StdTuneLR2[1][-1]
    finaltestlosses_tune[12] = StdTuneLR3[1][-1]
    finaltestlosses_tune[13] = StdTuneLR4[1][-1]
    finaltestlosses_tune[14] = StdTuneLR5[1][-1]
    finaltestlosses_tune[15] = StdTuneLR6[1][-1]
    finaltestlosses_tune[16] = StdTuneLR7[1][-1]
    finaltestlosses_tune[17] = StdTuneLR8[1][-1]
    finaltestlosses_tune[18] = StdTuneLR9[1][-1]
    finaltestlosses_tune[19] = StdTuneLR10[1][-1]

    finaltestlosses_tune[20] = LargeTuneET1[1][-1]
    finaltestlosses_tune[21] = LargeTuneET2[1][-1]
    finaltestlosses_tune[22] = LargeTuneET3[1][-1]
    finaltestlosses_tune[23] = LargeTuneET4[1][-1]
    finaltestlosses_tune[24] = LargeTuneET5[1][-1]
    finaltestlosses_tune[25] = LargeTuneET6[1][-1]
    finaltestlosses_tune[26] = LargeTuneET7[1][-1]
    finaltestlosses_tune[27] = LargeTuneET8[1][-1]
    finaltestlosses_tune[28] = LargeTuneET9[1][-1]
    finaltestlosses_tune[29] = LargeTuneET10[1][-1]

    finaltestlosses_tune[30] = LargeTuneLR1[1][-1]
    finaltestlosses_tune[31] = LargeTuneLR2[1][-1]
    finaltestlosses_tune[32] = LargeTuneLR3[1][-1]
    finaltestlosses_tune[33] = LargeTuneLR4[1][-1]
    finaltestlosses_tune[34] = LargeTuneLR5[1][-1]
    finaltestlosses_tune[35] = LargeTuneLR6[1][-1]
    finaltestlosses_tune[36] = LargeTuneLR7[1][-1]
    finaltestlosses_tune[37] = LargeTuneLR8[1][-1]
    finaltestlosses_tune[38] = LargeTuneLR9[1][-1]
    finaltestlosses_tune[39] = LargeTuneLR10[1][-1]

    corr2 = np.corrcoef(silhouette_scores, finaltestlosses_tune)[0, 1]
    finaltestlosses_tune_1 = np.zeros(20)
    finaltestlosses_tune_1[:10] = finaltestlosses_tune[:10]
    finaltestlosses_tune_1[10:20] = finaltestlosses_tune[20:30]
    finaltestlosses_tune_2 = np.zeros(20)
    finaltestlosses_tune_2[:10] = finaltestlosses_tune[10:20]
    finaltestlosses_tune_2[10:20] = finaltestlosses_tune[30:40]
    corr2_1 = np.corrcoef(silhouette_scores_1, finaltestlosses_tune_1)[0, 1]
    corr2_2 = np.corrcoef(silhouette_scores_2, finaltestlosses_tune_2)[0, 1]


    delta_silhouette_scores = np.zeros(20)
    # Original is 40 in 4 block sof 10: StdET, StdLR, LargeET, LargeLR
    delta_silhouette_scores[:10] = silhouette_scores[:10] - silhouette_scores[20:30]
    delta_silhouette_scores[10:] = silhouette_scores[10:20] - silhouette_scores[30:]
    delta_silhouette_scores_1 = delta_silhouette_scores[:10]
    delta_silhouette_scores_2 = delta_silhouette_scores[10:]
    delta_finaltestlosses_train = np.zeros(20)
    delta_finaltestlosses_train[:10] = finaltestlosses_train[:10] - finaltestlosses_train[20:30]
    delta_finaltestlosses_train[10:] = finaltestlosses_train[10:20] - finaltestlosses_train[30:]
    delta_finaltestlosses_train_1 = delta_finaltestlosses_train[:10]
    delta_finaltestlosses_train_2 = delta_finaltestlosses_train[10:]
    delta_finaltestlosses_tune = np.zeros(20)
    delta_finaltestlosses_tune[:10] = finaltestlosses_tune[:10] - finaltestlosses_tune[20:30]
    delta_finaltestlosses_tune[10:] = finaltestlosses_tune[10:20] - finaltestlosses_tune[30:]
    delta_finaltestlosses_tune_1 = delta_finaltestlosses_tune[:10]
    delta_finaltestlosses_tune_2 = delta_finaltestlosses_tune[10:]

    delta_corr1 = np.corrcoef(delta_silhouette_scores, delta_finaltestlosses_train)[0, 1]
    delta_corr1_1 = np.corrcoef(delta_silhouette_scores_1, delta_finaltestlosses_train_1)[0, 1]
    delta_corr1_2 = np.corrcoef(delta_silhouette_scores_2, delta_finaltestlosses_train_2)[0, 1]

    delta_corr2 = np.corrcoef(delta_silhouette_scores, delta_finaltestlosses_tune)[0, 1]
    delta_corr2_1 = np.corrcoef(delta_silhouette_scores_1, delta_finaltestlosses_tune_1)[0, 1]
    delta_corr2_2 = np.corrcoef(delta_silhouette_scores_2, delta_finaltestlosses_tune_2)[0, 1]

    # Once again, oiginal is 40 in 4 blocks of 10: StdET, StdLR, LargeET, LargeLR

    delta2_finaltestlosses_train = np.zeros(40)
    delta2_finaltestlosses_train[:10] = finaltestlosses_train[:10] - finaltestlosses_train[20:30]
    delta2_finaltestlosses_train[10:20] = finaltestlosses_train[10:20] - finaltestlosses_train[30:]
    delta2_finaltestlosses_train[20:30] = finaltestlosses_train[20:30] - finaltestlosses_train[:10]
    delta2_finaltestlosses_train[30:] = finaltestlosses_train[30:] - finaltestlosses_train[10:20]
    delta2_finaltestlosses_train_1 = np.zeros(20)
    delta2_finaltestlosses_train_1[:10] = delta2_finaltestlosses_train[:10]
    delta2_finaltestlosses_train_1[10:] = delta2_finaltestlosses_train[20:30]
    delta2_finaltestlosses_train_2 = np.zeros(20)
    delta2_finaltestlosses_train_2[:10] = delta2_finaltestlosses_train[10:20]
    delta2_finaltestlosses_train_2[10:] = delta2_finaltestlosses_train[30:]

    delta2_finaltestlosses_tune = np.zeros(40)
    delta2_finaltestlosses_tune[:10] = finaltestlosses_tune[:10] - finaltestlosses_tune[20:30]
    delta2_finaltestlosses_tune[10:20] = finaltestlosses_tune[10:20] - finaltestlosses_tune[30:]
    delta2_finaltestlosses_tune[20:30] = finaltestlosses_tune[20:30] - finaltestlosses_tune[:10]
    delta2_finaltestlosses_tune[30:] = finaltestlosses_tune[30:] - finaltestlosses_tune[10:20]
    delta2_finaltestlosses_tune_1 = np.zeros(20)
    delta2_finaltestlosses_tune_1[:10] = delta2_finaltestlosses_tune[:10]
    delta2_finaltestlosses_tune_1[10:] = delta2_finaltestlosses_tune[20:30]
    delta2_finaltestlosses_tune_2 = np.zeros(20)
    delta2_finaltestlosses_tune_2[:10] = delta2_finaltestlosses_tune[10:20]
    delta2_finaltestlosses_tune_2[10:] = delta2_finaltestlosses_tune[30:]

    delta2_corr1 = np.corrcoef(silhouette_scores, delta2_finaltestlosses_train)[0, 1]
    delta2_corr1_1 = np.corrcoef(silhouette_scores_1, delta2_finaltestlosses_train_1)[0, 1]
    delta2_corr1_2 = np.corrcoef(silhouette_scores_2, delta2_finaltestlosses_train_2)[0, 1]

    delta2_corr2 = np.corrcoef(silhouette_scores, delta2_finaltestlosses_tune)[0, 1]
    delta2_corr2_1 = np.corrcoef(silhouette_scores_1, delta2_finaltestlosses_tune_1)[0, 1]
    delta2_corr2_2 = np.corrcoef(silhouette_scores_2, delta2_finaltestlosses_tune_2)[0, 1]


    # New Cohen Request - show a small version of all 10 trials of the ET variant. Training curves only
    fig, ax = plt.subplots(2, 5, figsize=(20, 10))
    ax[0][0].semilogy(StdTrainET1[1])
    ax[0][0].semilogy(LargeTrainET1[1])
    ax[0][1].semilogy(StdTrainET2[1])
    ax[0][1].semilogy(LargeTrainET2[1])
    ax[0][2].semilogy(StdTrainET3[1])
    ax[0][2].semilogy(LargeTrainET3[1])
    ax[0][3].semilogy(StdTrainET4[1])
    ax[0][3].semilogy(LargeTrainET4[1])
    ax[0][4].semilogy(StdTrainET5[1])
    ax[0][4].semilogy(LargeTrainET5[1])
    ax[1][0].semilogy(StdTrainET6[1])
    ax[1][0].semilogy(LargeTrainET6[1])
    ax[1][1].semilogy(StdTrainET7[1])
    ax[1][1].semilogy(LargeTrainET7[1])
    ax[1][2].semilogy(StdTrainET8[1])
    ax[1][2].semilogy(LargeTrainET8[1])
    ax[1][3].semilogy(StdTrainET9[1])
    ax[1][3].semilogy(LargeTrainET9[1])
    ax[1][4].semilogy(StdTrainET10[1])
    ax[1][4].semilogy(LargeTrainET10[1])
    ax[0][0].set_ylim([1e0, 1e3])
    ax[0][0].set_ylim([1e0, 1e3])
    ax[0][1].set_ylim([1e0, 1e3])
    ax[0][2].set_ylim([1e0, 1e3])
    ax[0][3].set_ylim([1e0, 1e3])
    ax[0][4].set_ylim([1e0, 1e3])
    ax[1][0].set_ylim([1e0, 1e3])
    ax[1][1].set_ylim([1e0, 1e3])
    ax[1][2].set_ylim([1e0, 1e3])
    ax[1][3].set_ylim([1e0, 1e3])
    ax[1][4].set_ylim([1e0, 1e3])
    ax[0][0].legend(['Std','Large'])
    ax[0][1].legend(['Std', 'Large'])
    ax[0][2].legend(['Std', 'Large'])
    ax[0][3].legend(['Std', 'Large'])
    ax[0][4].legend(['Std', 'Large'])
    ax[1][0].legend(['Std', 'Large'])
    ax[1][1].legend(['Std', 'Large'])
    ax[1][2].legend(['Std', 'Large'])
    ax[1][3].legend(['Std', 'Large'])
    ax[1][4].legend(['Std', 'Large'])
    ax[0][0].set_xlabel('Iteration')
    ax[0][1].set_xlabel('Iteration')
    ax[0][2].set_xlabel('Iteration')
    ax[0][3].set_xlabel('Iteration')
    ax[0][4].set_xlabel('Iteration')
    ax[1][0].set_xlabel('Iteration')
    ax[1][1].set_xlabel('Iteration')
    ax[1][2].set_xlabel('Iteration')
    ax[1][3].set_xlabel('Iteration')
    ax[1][4].set_xlabel('Iteration')
    ax[0][0].set_ylabel('Generalization Loss')
    ax[0][1].set_ylabel('Generalization Loss')
    ax[0][2].set_ylabel('Generalization Loss')
    ax[0][3].set_ylabel('Generalization Loss')
    ax[0][4].set_ylabel('Generalization Loss')
    ax[1][0].set_ylabel('Generalization Loss')
    ax[1][1].set_ylabel('Generalization Loss')
    ax[1][2].set_ylabel('Generalization Loss')
    ax[1][3].set_ylabel('Generalization Loss')
    ax[1][4].set_ylabel('Generalization Loss')
    ax[0][0].set_title('Single Task Training Trail 1')
    ax[0][1].set_title('Single Task Training Trail 2')
    ax[0][2].set_title('Single Task Training Trail 3')
    ax[0][3].set_title('Single Task Training Trail 4')
    ax[0][4].set_title('Single Task Training Trail 5')
    ax[1][0].set_title('Single Task Training Trail 6')
    ax[1][1].set_title('Single Task Training Trail 7')
    ax[1][2].set_title('Single Task Training Trail 8')
    ax[1][3].set_title('Single Task Training Trail 9')
    ax[1][4].set_title('Single Task Training Trail 10')
    fig.suptitle('Test')
    plt.show()