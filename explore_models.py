#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:33:34 2020

@author: simone
"""

#%%
import torch
from AAECG_2 import CBeatAAE
import Utils as u
from Ecgdataset import Ecgdataset
import Baselines as b
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
from random import randint
#%% load models

aae = CBeatAAE(nz = 2)
aae.load_model()

ae = b.AE()
ae.load_model()

fAnoGan = b.FAnoGan(nz = 5)
fAnoGan.load_model("models/f_anoGAN/")

pca = b.ADPCA()
pca.load_model()

#%%
# Load the dataset
dataset = Ecgdataset("intra/abnormal/", upload_on_ram = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle= True)

#%%
""" 3 men and 3 women """
label = torch.tensor([[1,0,0],
                      [1,0,0],
                      [1,0,0],
                      [0,1,0],
                      [0,1,0],
                      [0,1,0]])

z = aae.sample_noise(len(label))
beat = aae.netD(z,label).detach().numpy()
u.plot_some_beat([beat[i] for i in range(len(label))])


z = fAnoGan.sample_noise(len(label))
beat = fAnoGan.netD(z).detach().numpy()
u.plot_some_beat([beat[i] for i in range(len(label))])



#%%

# plot a real beat over its reconstruction


for i,load in enumerate(dataloader):
    beat = load[0][:1]
    Dx = load[2]['label'][0]
    typ = load[2]['diagnosys'][0]
    if typ != 'Q':
        continue
    # AAE
    reconstructed = aae.netD(aae.netE(beat), load[1][:1]).detach()
    pred = aae.get_anomaly_score(beat, load[1][:1]) > aae.threeshold
    pred = "Abnormal" if pred else "Normal"
    plt.figure()
    plt.title("MyModel: "+Dx+" predicted as "+pred)
    plt.plot(beat.view(-1).numpy(), label='original')
    plt.plot(reconstructed.view(-1).detach().numpy(), label = 'recontruction')
    plt.legend()
    
    # AE
    reconstructed = ae.netD(ae.netE(beat)).detach()
    pred = ae.get_anomaly_score(beat) > ae.threeshold
    pred = "Abnormal" if pred else "Normal"
    plt.figure()
    plt.title("AutoEnc: "+Dx+" predicted as "+pred)
    plt.plot(beat.view(-1).numpy(), label='original')
    plt.plot(reconstructed.view(-1).detach().numpy(), label = 'recontruction')
    plt.legend()
    
    # FCANOGAN
    reconstructed = fAnoGan.netD(fAnoGan.netE(beat)).detach()
    pred = fAnoGan.get_anomaly_score(beat) > fAnoGan.threeshold
    pred = "Abnormal" if pred else "Normal"
    plt.figure()
    plt.title("fAnoGan: "+Dx+" predicted as "+pred)
    plt.plot(beat.view(-1).numpy(), label='original')
    plt.plot(reconstructed.view(-1).detach().numpy(), label = 'recontruction')
    plt.legend()
    
    # PCA
    X = beat[0].numpy()
    pca.plot_rec(X, label= Dx)
    print("diagnosys is "+typ)
    break
    
    
"""
EXPLORE THE LATENT SPACE
"""
    
#%%
# verify if it is uniform
b_s = 512
dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_s)
load = next(iter(dataloader))
beats = load[0]
label = load[2]['label']
latent = aae.netE(beats).detach().numpy()
unif = aae.sample_noise(b_s).numpy()
plt.figure()
plt.scatter(latent[:, 0], latent[:, 1], label='latents', cmap=label)
plt.scatter(unif[:,0], unif[:,1], label = 'uniform')
plt.legend()

#%%
# abnormal and normal latent codes

normal = Ecgdataset("intra/normal/")
normal_dl = torch.utils.data.DataLoader(normal, batch_size=512, shuffle= True)
abnormal = Ecgdataset("intra/abnormal/")
abnormal_dl = torch.utils.data.DataLoader(abnormal, batch_size=512)

load = next(iter(dataloader))
beats = load[0]
label = load[2]['label']
latent = aae.netE(beats).detach().numpy()

colors = ['r' if l == 'abnormal' else 'b' for l in label]
plt.figure()
plt.title('latent per label')
plt.scatter(latent[:, 0], latent[:, 1], c=colors)
plt.legend()

#%%
# visualize false positive
normal = Ecgdataset("intra/normal/")
normal_dl = torch.utils.data.DataLoader(normal, batch_size=1000, shuffle= True)

load = next(iter(normal_dl))
beat = load[0]
pred = ae.get_anomaly_score(beat) > ae.threeshold

false_positives = beat[np.where(pred)]

for idx in range(false_positives.size(0)):
    fbeat = false_positives[idx:(idx+1)]
    plt.figure()
    plt.title("AutoEncoder")
    reconstructed = ae.netD(ae.netE(fbeat)).detach()
    plt.plot(fbeat.view(-1).numpy(), label='original')
    plt.plot(reconstructed.view(-1).detach().numpy(), label = 'recontruction')
    plt.legend()

#%%
# visualize false negative

abnormal = Ecgdataset("intra/abnormal/")
abnormal_dl = torch.utils.data.DataLoader(abnormal, batch_size=1000)

load = next(iter(abnormal_dl))
beat = load[0]
asc = ae.get_anomaly_score(beat)
pred = asc < ae.threeshold

false_negatives = beat[np.where(pred)]

for idx in range(min(false_negatives.size(0), 5)):
    fbeat = false_negatives[idx:(idx+1)]
    diag = load[2]['diagnosys'][np.where(pred)[0][idx]]
    plt.figure()
    plt.title("AutoEncoder "+diag)
    reconstructed = ae.netD(ae.netE(fbeat)).detach()
    plt.plot(fbeat.view(-1).numpy(), label='original')
    plt.plot(reconstructed.view(-1).detach().numpy(), label = 'recontruction')
    plt.legend()





#%%
# visualize false positive pca

normal = Ecgdataset("intra/normal/")
normal_dl = torch.utils.data.DataLoader(normal, batch_size=5000, shuffle= True)

load = next(iter(normal_dl))
beat = load[0].numpy()

pred = pca.get_anomaly_score(beat) > pca.threeshold

false_positives = beat[np.where(pred)]

for idx in range(min(false_positives.shape[0], 5)):
    fbeat = false_positives[idx:(idx+1)]
    pca.plot_rec(fbeat)
#%%
# visualize false negative pca

normal = Ecgdataset("intra/abnormal/")
normal_dl = torch.utils.data.DataLoader(normal, batch_size=5000, shuffle= True)

load = next(iter(normal_dl))
beat = load[0].numpy()

pred = pca.get_anomaly_score(beat) < pca.threeshold

false_positives = beat[np.where(pred)]

for idx in range(min(false_positives.shape[0], 5)):
    fbeat = false_positives[idx:(idx+1)]
    pca.plot_rec(fbeat)


#%%
# false negative AAE

abnormal = Ecgdataset("intra/abnormal/")
abnormal_dl = torch.utils.data.DataLoader(abnormal, batch_size=1000)

load = next(iter(abnormal_dl))
label = load[1]
beat = load[0]
asc = aae.get_anomaly_score(beat, label)
pred = asc < aae.threeshold

false_negatives = beat[np.where(pred)]
fn_label = label[np.where(pred)]

for idx in range(min(false_negatives.size(0), 20)):
    fbeat = false_negatives[idx:(idx+1)]
    flabel = fn_label[idx:(idx+1)]
    diag = load[2]['diagnosys'][np.where(pred)[0][idx]]
    plt.figure()
    plt.title("AutoEncoder "+diag)
    reconstructed = aae.netD(aae.netE(fbeat), flabel).detach()
    plt.plot(fbeat.view(-1).numpy(), label='original')
    plt.plot(reconstructed.view(-1).detach().numpy(), label = 'recontruction')
    plt.legend()

#%%
# code exploration sex differences

# [0.5236, -0.4329, -0.5555, 1.7917, 0.4552] è bello 

label = torch.tensor([[1,0,0],# female
                      [0,1,0]])# male
n = 5

fig, axes = plt.subplots(2,n, sharex=True, sharey=True)

fig.suptitle('Sex differences between ECG with same code', fontsize=15)

for i in range(n):
    z = aae.sample_noise(1).repeat((2,1))
    beats = aae.netD(z,label).detach().numpy().squeeze()
    for j in range(2):
        # generated a heartbeat
        if i == 0:
            sex = 'female' if j ==0 else 'male'
            axes[j, i].set_ylabel(sex)
        if j == 0:
            axes[j, i].set_title('latent code: \n'+str(np.round(z[0].numpy(), 2)), fontsize= 8)
        axes[j,i].plot(beats[j])
        
plt.show()
#%%
# origin beat
label = torch.tensor([[1,0,0],# female
                      [0,1,0]])# male
z = torch.Tensor([[0,0,0,0,0],[0,0,0,0,0]])
beat = aae.netD(z,label).detach().numpy()
u.plot_some_beat([beat[i] for i in range(len(label))],  titles_list=['Female', 'Male'])

#%%
# variance of the encoder vs anomaly score
normal = Ecgdataset("intra/normal/")
normal_dl = torch.utils.data.DataLoader(normal, batch_size=2000, shuffle= True)
nl = next(iter(normal_dl))
abnormal = Ecgdataset("intra/abnormal/")
abnormal_dl = torch.utils.data.DataLoader(abnormal, batch_size=2000)
al = next(iter(abnormal_dl))


beats = torch.cat((nl[0], al[0]), dim = 0)
labels = [*nl[2]['label'], *al[2]['label']]
codes = torch.cat((nl[1], al[1]), dim = 0)

variances = np.log(1+np.max(aae.netE.variance(beats).detach().numpy(), axis = 1))
an_sc = aae.get_anomaly_score(beats,codes)

import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color='red', label='abnormal')
blue_patch = mpatches.Patch(color='blue', label='normal')


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(an_sc.reshape(-1,1), variances)

x = np.arange(0,1, 0.001)
y = reg.coef_ * x + reg.intercept_
fig, ax = plt.subplots()
col = np.array(labels) == 'abnormal'
ax.scatter(an_sc, variances, s = 0.9, c = [('r' if i== 1 else 'b') for i in col])
ax.plot(x,y, c = 'g', label="linear regression line", linewidth=0.7)
ax.set_xlabel("anomaly score")
ax.set_ylabel("encoder log-variance")
ax.set_title("")
handles, _ = ax.get_legend_handles_labels()
handles.extend([red_patch, blue_patch]) 
plt.legend(handles = handles)
plt.show()

from statsmodels.api import OLS
OLS(variances,an_sc.reshape(-1,1)).fit().summary()



#%%
# hidden code modification
import matplotlib.colors as colors
import matplotlib.cm as cmx

label = torch.tensor([[1,0,0]])

n = 256
gamma = 10

values = np.arange(gamma* (0-n/2)/n, gamma* (n-n/2)/n, step = gamma/n)
plt.figure()

hidden_code_to_modify = 2
z = torch.Tensor([[0.5236, -0.4329, -0.5555, 1.7917, 0.4552]])
z = torch.Tensor([[0,0,0,0,0]])
jet = cm = plt.get_cmap('winter') 
cNorm  = colors.Normalize(vmin=values[0], vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
for i in range(n):
    z_m = z
    z_m[:,hidden_code_to_modify] = gamma* (i-n/2)/n
    beat = aae.netD(z_m,label).detach().numpy().squeeze()
    colorVal = scalarMap.to_rgba(values[i])
    plt.plot(beat, label = "code "+str((i-n/2)/n), color = colorVal)

plt.colorbar(scalarMap)
plt.title("Varying the h"+str(hidden_code_to_modify+1)+" fixing other to zero")
#plt.legend()
plt.show()



#%%
# max min variation in the interval

plt.figure()
for i in range(2):
    z_m = z
    z_m[:,hidden_code_to_modify] = gamma* (n*i-n/2)/n
    beat = aae.netD(z_m,label).detach().numpy().squeeze()
    plt.plot(beat, label = "code "+str((i-n/2)/n))
#plt.legend()
plt.show()
                      
#%%
# visualize the manifold of normal heartbeats
# nxn grid of generated heartbeats
n = 10

fig, axes = plt.subplots(n, n, sharex=True, sharey=True)

fig.suptitle('Manifold of normal heartbeats', fontsize=15)


z = torch.Tensor([[0.5236, -0.4329, -0.5555, 1.7917, 0.4552]])
label = torch.tensor([[1,0,0]])
gamma = 10

for i in range(n):
    for j in range(n):
        # generated a heartbeat
        z_m = z
        z_m[:,:2] = torch.Tensor([gamma* (i-n/2)/n, gamma* (j-n/2)/n])
        beat = aae.netD(z_m,label).detach().numpy().squeeze()
        axes[i,j].plot(beat)
        axes[i,j].set_axis_off()

fig.text(0.5, 0.04, 'Varying first hidden code between '+str(gamma* (0-n/2)/n)+" and "+str(gamma* ((n)-n/2)/n), ha='center')
fig.text(0.04, 0.5, 'Varying second hidden code between '+str(gamma* (0-n/2)/n)+" and "+str(gamma* ((n)-n/2)/n), va='center', rotation='vertical')

#%%

#%%
# visualize the manifold of normal heartbeats
# nxn grid of generated heartbeats
n = 10

fig, axes = plt.subplots(n, n, sharex=True, sharey=True)

fig.suptitle('Manifold of normal heartbeats', fontsize=15)


z =  torch.Tensor([[0.5236, -0.4329, -0.5555, 1.7917, 0.4552]])
z_orthogonal = torch.Tensor([[-0.5236, 0.4329, 0.5555, 1.7917, -0.4552]])
label = torch.tensor([[1,0,0]]) # fix a male
gamma = 2
beta = 2

for i in range(n):
    for j in range(n):
        # generated a heartbeat
        z_m = z*i*gamma + z_orthogonal*j*beta
        beat = aae.netD(z_m,label).detach().numpy().squeeze()
        axes[i,j].plot(beat)
        axes[i,j].set_axis_off()

#fig.text(0.5, 0.04, 'Varying hidden codes between '+str(gamma* (0-n/2)/n)+" and "+str(gamma* ((n)-n/2)/n), ha='center')
#fig.text(0.04, 0.5, 'Varying second hidden code between '+str(gamma* (0-n/2)/n)+" and "+str(gamma* ((n)-n/2)/n), va='center', rotation='vertical')





#%%

    
    







