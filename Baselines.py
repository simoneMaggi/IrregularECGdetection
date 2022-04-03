#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:29:14 2021

@author: simone
"""

import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, fbeta_score
from tqdm import tqdm
from Utils import plot_some_beat
import pickle
num_cores = multiprocessing.cpu_count()
import os
#%%
def plot_error_mixing(errors, labels, best_thr, est_thr = None):
    plt.figure()
    h = np.histogram(errors[np.where(labels == 0)])
    plt.bar(h[1][:-1], h[0]/np.sum(h[0]), label = 'normal', width = 0.001)
    h = np.histogram(errors[np.where(labels == 1)])
    plt.bar( h[1][:-1], h[0]/np.sum(h[0]), label = 'abnormal', width = 0.001)
    plt.ylim([0, 1.001])
    plt.xlim([0, np.max(errors)])
    if est_thr is None:
        plt.vlines(x = best_thr, ymin = 0, ymax = 1, colors = 'b', label = 'threeshold')
    else:
        plt.vlines(x = est_thr, ymin = 0, ymax = 1, colors = 'b', label = 'alpha-thr')
        plt.vlines(x = best_thr, ymin = 0, ymax = 1, colors = 'r', label = 'best f1-thr')
    plt.legend()
    plt.show()
        
def get_threeshold(errors, labels = None, alpha = 0.05, verbose = False):
    # Estimate threeshold only with normal data
    # fixing false positive rate to alpha.
    # Then, if provided, use the labels to compute the threeshold
    # which gives the best recall ( = accuracy)
    # if the threeshold is greater than the previously computed
    # that means that we have decreased also the false positive rate
    # Hence we select the new threeshold
    
    if verbose:
        print("estimate threeshold with only normal data")
    if labels is None:
        ecdf = ECDF(errors)
    else:
        ecdf = ECDF(errors[np.where(labels == 0)])
    thr = ecdf.x[np.where(ecdf.y >= (1-alpha))[0][0]]
    quantile_thr = thr
    if labels is None:
        return thr
    best_thr = 0
    # Estimate threeshold with both
    thrs = np.linspace(np.min(errors), np.max(errors), num=errors.shape[0]*2)
    if verbose:
        print("estimate threeshold with all data")
    e = Parallel(n_jobs=num_cores)(delayed(fbeta_score)(errors >= t, labels, beta = 2)
                           for t in thrs)
    ind = np.argmax(np.array(e))
    best_thr = thrs[ind]
    if verbose:
        plot_error_mixing(errors, labels, best_thr, est_thr = thr)
    
    return quantile_thr, best_thr
    
class AE:
    def __init__(self,device = torch.device('cpu'), N=280, 
                 L=1, nz=50,nc=3,nef=32, ngpu = 0):
        # Lenght of a beat.
        self.N = N
        
        # Number of leads
        self.L = L
        
        # Size of z latent vector 
        self.nz = nz
        
        # Size of feature maps in encoder
        self.nef = nef
        
        self.device = device
        self.ngpu = ngpu
        
        self.netE = self.Encoder( L, N,nef, nz, ngpu) 
        self.netD = self.Decoder( nef, nz, nc, L, ngpu)
        
        self.netE.to(device)
        self.netD.to(device)
        
        self.initialize()
        
        self.trained = False
        
        # Initialize the threeshold
        self.threeshold = 0
        
    class Encoder(nn.Module):
        def __init__(self, L, N, nef, nz, ngpu):
            super(AE.Encoder, self).__init__()
            self.ngpu = ngpu
            self.conv = nn.Sequential(
                # input Beat dimensions L x N => L x N 
                nn.Conv1d( L, nef, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef),
                nn.LeakyReLU(True),
                # state size (nef) x 128 
                nn.Conv1d( nef, nef * 2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 2),
                nn.LeakyReLU(True),
                # state size. (nef*2) x 64
                nn.Conv1d(nef * 2, nef * 4, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 4),
                nn.LeakyReLU(True),
                # state size. (nef*4) x 32
                nn.Conv1d( nef * 4, nef * 8, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 8),
                nn.LeakyReLU(True),
                # state size. (nef*8) x 16
                nn.Conv1d( nef * 8, nef * 16, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef*16),
                nn.LeakyReLU(True),
                # state size. (nef * 16) x 8
                nn.Conv1d(nef * 16, nz, 8, 1, 0),
                nn.Flatten())
            
        def forward(self, x):
            return self.conv(x)
        
    class Decoder(nn.Module):
        def __init__(self, nef, nz, nc, L, ngpu):
            super(AE.Decoder, self).__init__()
            self.ngpu = ngpu
            self.upconv = nn.Sequential(
                # input nz x 1
                nn.ConvTranspose1d(nz, nef * 16, 8, 1, 0, bias = False),
                nn.BatchNorm1d(nef * 16),
                nn.ReLU(True),
                # state size (nef*16) x 8 
                nn.ConvTranspose1d( nef*16, nef * 8, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 8),
                nn.ReLU(True),
                # state size. (nef*8) x 16 
                nn.ConvTranspose1d(nef * 8, nef * 4, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 4),
                nn.ReLU(True),
                # state size. (nef*4) x 32 
                nn.ConvTranspose1d( nef * 4, nef * 2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 2),
                nn.ReLU(True),
                # state size. (nef*2) x 64 
                nn.ConvTranspose1d( nef * 2, nef, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef),
                nn.ReLU(True),
                # state size. (nef) x 128 
                nn.ConvTranspose1d( nef, L, 4, 2, 1, bias=False),
                # state size. (L) x 256 
            )
    
        def forward(self, z):
            return self.upconv(z.unsqueeze(2))
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def initialize(self):
        self.netE.apply(self.weights_init)
        self.netD.apply(self.weights_init)
    
    def get_anomaly_score(self, X):
        b_s = X.size(0) # Batch size
        X_rec = self.netD(self.netE(X)).view(b_s,-1).detach().cpu().numpy()
        X = X.view(b_s,-1).detach().cpu().numpy()
        AS = np.mean(np.square(X - X_rec), axis = 1)
        return AS
    
    def train(self,  train_dl, valid_dl, num_epochs, lr=0.0001, early_stop = False,
               beta1=0.5, models_folder = "models/AE/"):
        rec_losses = []
        example = next(iter(train_dl))
        examples_data = example[1][:4].to(self.device)
        beat_list = []
        
        MSE = nn.MSELoss().to(self.device)
        optimizerE = optim.Adam(self.netE.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        
        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            for i, data in enumerate(train_dl, 0):
                self.netE.zero_grad()
                self.netD.zero_grad()
                
                X = data[0].to(self.device)
                X_rec = self.netD(self.netE(X))
                
                ED_rec_loss = MSE(X, X_rec)
                ED_rec_loss.backward()
                optimizerD.step()
                optimizerE.step()
                
                ## Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\n\t Rec Loss: %.4f'
                          % (epoch, num_epochs, i, len(train_dl),
                             ED_rec_loss.item()))
                
                rec_losses.append(ED_rec_loss.item())
        
                
        torch.save(self.netE.state_dict(), models_folder+"encoder.mod")
        torch.save(self.netD.state_dict(), models_folder+"decoder.mod")
        pred = None
        labels = None
        print("\t Computing the optimal threeshold")
        pbar = tqdm.tqdm(total = len(valid_dl))
        print("\t Compute errors")
        for i,data in enumerate(valid_dl, 0):
            beats = data[0].to(self.device)
            label = np.array(data[2]['label']) == 'abnormal' # 0 normale 1 anormale
            ano_sc = self.get_anomaly_score(beats)
            if pred is None:
                pred = ano_sc
                labels = label
            else:
                pred = np.concatenate((pred, ano_sc), axis = 0)
                labels = np.concatenate((labels, label), axis = 0)
            pbar.update(n = 1)
        pbar.close()
        # Compute threeshold using both errors
        self.threeshold = get_threeshold(pred, labels = labels, verbose = True)
        
        # Save the best threeshold
        np.save(models_folder+"thr.npy", np.array([self.threeshold]))
        f, ax = plt.subplots()
        ax.set_title("Reconstruction error ")
        ax.plot(rec_losses)
        plt.xlabel("iterations")
        plt.show()
        for i in range(len(beat_list)):
            plot_some_beat(beat_list[i], suptitle = 'epoch: '+str(i), 
                             titles_list = [('M' if examples_data[j,0]==0 else 'F' ) for j in range(len(beat_list[i]))])
        self.trained = True
    
    def test(self, test_dl, result_folder= "models/AE/"):
        if not self.trained:
            print("train or load a model")
            return None
        print("Starting Testing Loop...")
        pred = None
        labels = None
        pbar = tqdm.tqdm(total=len(test_dl))
        for i,data in enumerate(test_dl, 0):
            beats = data[0].to(self.device)
            label = np.array(data[2]['label']) == 'abnormal' # 0 normale 1 anormale
            ano_sc = self.get_anomaly_score(beats)
            if pred is None:
                pred = ano_sc
                labels = label
            else:
                pred = np.concatenate((pred, ano_sc), axis = 0)
                labels = np.concatenate((labels, label), axis = 0)
            pbar.update(n=1)
        
        clrp = classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'])
        print("\n")
        print(clrp)
        plot_error_mixing(pred, labels, self.threeshold)
        rep = classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'],
                                         output_dict=True)
        auc = roc_auc_score(labels, pred)
        pr_auc = average_precision_score(labels, pred)
        result = {'rep':rep['abnormal'], 'roc_auc':auc, 'pr_auc':pr_auc,
                  'pr_curve':precision_recall_curve( labels, pred),
                  'f2score': fbeta_score(labels, pred >= self.threeshold, beta = 2)}
        
        with open(result_folder+'result.pickle', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'],
                                         output_dict=True)
    def load_model(self, model_folder = 'models/AE/'):
        self.netE.load_state_dict(torch.load(model_folder+"encoder.mod", map_location=self.device))
        self.netD.load_state_dict(torch.load(model_folder+"decoder.mod", map_location=self.device))
        # Load the threeshold
        self.threeshold = np.load(model_folder+"thr.npy")
        self.trained = True

#%%
from sklearn.decomposition import PCA
class ADPCA():
    def init(self):
        self.comp = None
        self.mean = None
        self.threeshold = 0
    
    def get_anomaly_score(self, X):
        X_gen = np.dot(np.dot(X.squeeze() - self.mean.squeeze(), self.comp.squeeze()), self.comp.T.squeeze()) + self.mean.squeeze()
        try:
            anomaly_score = np.mean(np.square(X_gen - X.squeeze()), axis = 1)
        except:
            anomaly_score = np.mean(np.square(X_gen - X))
        return anomaly_score
    
    
    def plot_rec(self, X, label = ''):
        pred = self.get_anomaly_score(X.squeeze()) > self.threeshold
        pred = "Abnormal" if pred else "Normal"
        plt.figure()
        plt.title("PCA "+label+" predicted as "+pred)
        plt.plot(X.squeeze(), label = 'original')
        rec = np.dot(np.dot(X.squeeze() - self.mean.squeeze(), self.comp.squeeze()), self.comp.T.squeeze()) + self.mean.squeeze()
        plt.plot(rec, label='reconstructed')
        plt.legend()
        plt.show()
    
    def train(self, dataset_train, dataset_valid, num_epochs = None, accounted_variance = 0.95, n_comp = 50):
        # fit pca on train
        pca = PCA(n_components=accounted_variance)
        pca.fit(dataset_train[:, 0:-1])
        self.mean = pca.mean_
        X = dataset_valid[:, :-1]
        y = dataset_valid[:, -1]
        self.comp = pca.components_.T
        print("\t Computing the optimal threeshold")
        errors = self.get_anomaly_score(X)
        _ , self.threeshold = get_threeshold(errors, labels = y, verbose = True)
        print("\t Save the model")
        np.save("models/PCA/thr.npy", np.array([self.threeshold]))
        np.save("models/PCA/comp.npy", np.array([self.comp]))
        np.save("models/PCA/mean.npy", np.array([self.mean]))
    
    def initialize(self):
        return
    def test(self, dataset_test, result_folder = 'models/PCA/'):
        if self.comp is None:
            print("\t Train or load the model")
            return
        X = dataset_test[:, :-1]
        labels = dataset_test[:, -1]
        
        pred = self.get_anomaly_score(X)
        print("shape "+str(pred.shape)+" "+str(labels.shape))
        clrp = classification_report(labels, pred >= self.threeshold, 
                                     target_names=['normal','abnormal'])
        print(clrp)
        plot_error_mixing(pred, labels, self.threeshold)
        rep = classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'],
                                         output_dict=True)
        auc = roc_auc_score(labels, pred)
        pr_auc = average_precision_score(labels, pred)
        result = {'rep':rep['abnormal'], 'roc_auc':auc, 'pr_auc':pr_auc,
                  'pr_curve':precision_recall_curve( labels, pred),
                  'f2score': fbeta_score(labels, pred >= self.threeshold, beta = 2)}
        with open(result_folder+'result.pickle', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'],
                                         output_dict=True)
    def load_model(self, model_folder = "models/PCA/"):
        self.threeshold = np.load(model_folder+"thr.npy")
        self.comp = np.load(model_folder+"comp.npy")
        self.mean = np.load(model_folder+"mean.npy")
        
        

#%%
class AnoBeat:
    def __init__(self,device = torch.device('cpu'), N=280, 
                 L=1, nz=50,nc=3,nef=32, ngpu = 0):
        # Lenght of a beat.
        self.N = N
        
        # Number of leads
        self.L = L
        
        # Size of z latent vector 
        self.nz = nz
        
        # Size of feature maps in encoder
        self.nef = nef
        
        self.device = device
        self.ngpu = ngpu
        
        self.netE = self.Encoder( L, N,nef, nz, ngpu) 
        self.netD = self.Decoder( nef, nz, nc, L, ngpu)
        self.netDis_visual = self.Discriminator_visual(L, N, nef, nz, ngpu)
        self.netDis_latent = self.Discriminator_latent(N, nz)
        
        self.netE.to(device)
        self.netD.to(device)
        self.netDis_latent.to(device)
        self.netDis_visual.to(device)
        
        # Initialize
        self.initialize()
        
        self.trained = False
        
        # Initialize the threeshold
        self.threeshold = 0
        
    class Encoder(nn.Module):
        def __init__(self, L, N, nef, nz, ngpu):
            super(AnoBeat.Encoder, self).__init__()
            self.ngpu = ngpu
            self.conv = nn.Sequential(
                # input Beat dimensions L x N => L x N 
                nn.Conv1d( L, nef, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace = True),
                # state size (nef) x 128 
                nn.Conv1d( nef, nef * 2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 2),
                nn.LeakyReLU(0.2, inplace = True),
                # state size. (nef*2) x 64
                nn.Conv1d(nef * 2, nef * 4, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 4),
                nn.LeakyReLU(0.2, inplace = True),
                # state size. (nef*4) x 32
                nn.Conv1d( nef * 4, nef * 8, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 8),
                nn.LeakyReLU(0.2, inplace = True),
                # state size. (nef*8) x 16
                nn.Conv1d( nef * 8, nef * 16, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef*16),
                nn.LeakyReLU(0.2, inplace = True),
                # state size. (nef * 16) x 8
                nn.Conv1d(nef * 16, nz, 8, 1, 0),
                nn.Flatten())
            
        def forward(self, x):
            return self.conv(x)
        
    class Decoder(nn.Module):
        def __init__(self, nef, nz, nc, L, ngpu):
            super(AnoBeat.Decoder, self).__init__()
            self.ngpu = ngpu
            self.upconv = nn.Sequential(
                # input nz x 1
                nn.ConvTranspose1d(nz, nef * 16, 8, 1, 0, bias = False),
                nn.BatchNorm1d(nef * 16),
                nn.ReLU(True),
                # state size (nef*16) x 8 
                nn.ConvTranspose1d( nef*16, nef * 8, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 8),
                nn.ReLU(True),
                # state size. (nef*8) x 16 
                nn.ConvTranspose1d(nef * 8, nef * 4, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 4),
                nn.ReLU(True),
                # state size. (nef*4) x 32 
                nn.ConvTranspose1d( nef * 4, nef * 2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 2),
                nn.ReLU(True),
                # state size. (nef*2) x 64 
                nn.ConvTranspose1d( nef * 2, nef, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef),
                nn.ReLU(True),
                # state size. (nef) x 128 
                nn.ConvTranspose1d( nef, L, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (L) x 256 
            )
    
        def forward(self, z):
            return self.upconv(z.unsqueeze(2))
        
    class Discriminator_visual(nn.Module):
        def __init__(self, L, N, nef, nz, ngpu):
            super(AnoBeat.Discriminator_visual, self).__init__()
            self.ngpu = ngpu
            self.critic = nn.Sequential(
                # input Beat dimensions L x N => L x 256 
                nn.Conv1d( L, nef, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # state size (nef) x 128 
                nn.Conv1d( nef, nef * 2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # state size. (nef*2) x 64
                nn.Conv1d(nef * 2, nef * 4, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 4),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # state size. (nef*4) x 32
                nn.Conv1d( nef * 4, nef * 8, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # state size. (nef*8) x 16
                nn.Conv1d( nef * 8, nef * 16, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef*16),
                nn.LeakyReLU(negative_slope=0.2, inplace=True))
                # state size. (nef * 16) x 8
            
            self.output = nn.Sequential(nn.Conv1d(nef * 16, 1, 8, 1, 0),
                nn.Flatten(),
                nn.Sigmoid())
        def features(self, x):
            return self.critic(x)
        
        def forward(self, x):
            return self.output(self.features(x))
    
    class Discriminator_latent(nn.Module):
        def __init__(self, N, nz, ndf = 4):
            super(AnoBeat.Discriminator_latent, self).__init__()
            self.critic = nn.Sequential(
                # input (1, 50)
                nn.Conv1d( 1, ndf, 4, 2, 0, bias=False),
                nn.BatchNorm1d(ndf),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # state size (ndf) x 24 
                nn.Conv1d( ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(ndf * 2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # state size. (ndf*2) x 12
                nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm1d(ndf * 4),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # state size. (ndf*4) x 6
                nn.Flatten()
                )
            self.output = nn.Sequential(nn.Linear(ndf*4*6, 1),
                                        nn.Sigmoid())
        def features(self, x):
            return self.critic(x.unsqueeze(1))
        
        def forward(self, x):
            return self.output(self.features(x))
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def get_anomaly_score(self, X):
        b_s = X.size(0) # Batch size
        X_rec = self.netD(self.netE(X)).view(b_s,-1).detach().cpu().numpy()
        X = X.view(b_s,-1).detach().cpu().numpy()
        AS = np.sum(np.square(X-X_rec), axis = 1)/self.N 
        return AS
    def initialize(self):
        self.netE.apply(self.weights_init)
        self.netD.apply(self.weights_init)
        self.netDis_latent.apply(self.weights_init)
        self.netDis_visual.apply(self.weights_init)
        
    def save(self, folder = '/'):
        torch.save(self.netDis_latent.state_dict(), folder+"discriminator_latent.mod")
        torch.save(self.netDis_visual.state_dict(), folder+"discriminator_visual.mod")
        torch.save(self.netE.state_dict(), folder+"encoder.mod")
        torch.save(self.netD.state_dict(), folder+"decoder.mod")
        np.save( folder+"thr.npy", self.threeshold)
        
        
    def train(self,  train_dl, valid_dl, num_epochs, lr = 0.0001, beta1=0.5, beta2 = 0.999, lambda1 = 1,
              lambda2 = 1, lambda3= 1, lambda4 = 1, model_folder = "models/AnoBeat/"):

        MSE_losses = []
        Disc_visual_losses = []
        
        ## Initialize Loss functions
        MSE = nn.MSELoss().to(self.device)
        BCE = nn.BCELoss().to(self.device)
        
        # Setup Adam optimizers for both Enc, Dec and Disc 
        optimizerE = optim.Adam(self.netE.parameters(), lr=lr, betas = (beta1, beta2))
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas = (beta1, beta2))
        optimizerDis_latent = optim.Adam(self.netDis_latent.parameters(), lr=lr, betas = (beta1, beta2))
        optimizerDis_visual = optim.Adam(self.netDis_visual.parameters(), lr= lr, betas=(beta1, beta2))
        
        best_metric = 0
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        
        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            try:
                for i, data in enumerate(train_dl, 0):
                    
                    # Format batch
                    X = data[0].to(self.device)
                    b_size = X.size(0)
                    
                    label_real = torch.ones(b_size).to(self.device)
                    label_fake = torch.zeros(b_size).to(self.device)
                    
                    # Update Visual Discriminator
                    self.netDis_visual.zero_grad()
                    
                    Z = self.netE(X)
                    X_rec = self.netD(Z)
                    #only real
                    D_real = self.netDis_visual(X)
                    D_real_loss = BCE(D_real.view(-1), label_real)
                    
                    # only fake
                    D_fake = self.netDis_visual(X_rec.detach())
                    D_fake_loss = BCE(D_fake.view(-1), label_fake)
                    
                    D_loss = D_fake_loss + D_real_loss
                    
                    if D_loss.item() < 5e-6:
                        print("\n\t re-initialize the discriminator ")
                        self.netDis_visual.apply(self.weights_init)
                    else:
                        D_loss.backward()
                        # Step
                        optimizerDis_visual.step()
                    
                    # Update Latent Discriminator
                    self.netDis_latent.zero_grad()
                    Z_rec = self.netE(X_rec).detach()
                    D_latent_loss = (BCE(self.netDis_latent(Z.detach()).view(-1), label_real) + 
                                     BCE(self.netDis_latent(Z_rec).view(-1), label_fake))
                    D_latent_loss.backward()
                    optimizerDis_latent.step()
                    
                    # Update Encoder and decoder
                    self.netD.zero_grad()
                    self.netE.zero_grad()
                    rec_loss = lambda1 * MSE(X, X_rec)
                    latent_loss = lambda2 * MSE(self.netDis_latent.features(Z),
                                                self.netDis_latent.features(Z_rec))
                    Ladvz = rec_loss + latent_loss
                    Ladvz.backward()
                    optimizerD.step()
                    optimizerE.step()
                    
                    # Update Decoder
                    self.netD.zero_grad()
                    noise = torch.randn((b_size,self.L, self.N), device = self.device)
                    X_noise = X + noise
                    X_rec = self.netD(self.netE(X))
                    r = MSE(X, X_rec)
                    Ladvn = MSE(self.netDis_visual.features(X),
                                self.netDis_visual.features(X_noise))
                    Ladvx = MSE(self.netDis_visual.features(X),
                                self.netDis_visual.features(X_rec))
                    Ladv = (lambda1 * r  - lambda3 * Ladvn + lambda4 * Ladvx)
                    Ladv.backward()
                    optimizerD.step()
                        
                    ## Output training stats
                    ## Output training stats
                    if i % 50 == 0:
                        print('[%d/%d][%d/%d]\n\t Discriminator visible Losses: real %.4f - fake %.4f \n\t  rec loss %.4f \n\t Discr latent loss: %.4f \n\t ladvn - ladvx: %.4f - %.4f'
                              % (epoch, num_epochs, i, len(train_dl),
                                 D_real_loss.item(), D_fake_loss.item() ,  rec_loss.item(), D_latent_loss.item(), Ladvn.item(), Ladvx.item()))
                        
                    # Save Losses for plotting later
                    MSE_losses.append(rec_loss.item())
                    Disc_visual_losses.append(D_real_loss.item() + D_fake_loss.item())
            except KeyboardInterrupt:
                print("\n INTERRUPT DETECTED: training stopped")
                break
            # evaluate and save best model
            print("\n Evaluate")
            self.compute_threeshold(valid_dl, optimal=True)
            labels, pred = self.evaluate(valid_dl)
            metric = average_precision_score(labels, pred)
            if metric >= best_metric:
                print("\n %.4f -----> %.4f "%(best_metric, metric))
                best_metric = metric
                self.save(model_folder)
            else:
                print("\n %.4f less than best %.4f"%(metric, best_metric))
        
        self.load_model(model_folder)
        f, ax = plt.subplots(2, 1, sharex = True)
        ax[0].set_title("Reconstruction Losses ")
        ax[0].plot(MSE_losses, label='mse error')
        ax[0].legend()
        ax[1].set_title("Discriminator losses")
        ax[1].plot(Disc_visual_losses, label = 'visual')
        ax[1].legend()
        ax[1].set_ylabel("Loss")
        plt.xlabel("iterations")
        plt.show()
        
        self.trained = True
    
    def evaluate(self, dl, verbose = False):
        pred = None
        labels = None
        if verbose:
            pbar = tqdm(total=len(dl))
        for i,data in enumerate(dl, 0):
            beats = data[0].to(self.device)
            label = np.array(data[2]['label']) == 'abnormal' # 0 normale 1 anormale
            ano_sc = self.get_anomaly_score(beats)
            if pred is None:
                pred = ano_sc
                labels = label
            else:
                pred = np.concatenate((pred, ano_sc), axis = 0)
                labels = np.concatenate((labels, label), axis = 0)
            if verbose:
                pbar.update(n=1)
        if verbose:
            pbar.close()
        return labels, pred
    
    def compute_threeshold(self, dl, optimal = False, verbose = False):
        labels, pred = self.evaluate(dl, verbose = verbose)
        # Compute threeshold using both errors
        if optimal:
            quantile_thr, best_thr = get_threeshold(pred, labels = labels, verbose = verbose)
            self.threeshold = best_thr
        else:
            quantile_thr, _ = get_threeshold(pred, labels = None, verbose = verbose)
            self.threeshold = quantile_thr
    
    def test(self, test_dl, result_folder='models/AnoBeat/'):
        if not self.trained:
            print("train or load a model")
            return None
        print("Starting Testing Loop...")
        labels,pred = self.evaluate(test_dl, verbose = True)
        
        clrp = classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'])
        print("\n")
        print(clrp)
        plot_error_mixing(pred, labels, self.threeshold)
        rep = classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'],
                                         output_dict=True)
        auc = roc_auc_score(labels, pred)
        pr_auc = average_precision_score(labels, pred)
        result = {'rep':rep['abnormal'], 'roc_auc':auc, 'pr_auc':pr_auc,
                  'pr_curve':precision_recall_curve( labels, pred),
                  'f2score': fbeta_score(labels, pred >= self.threeshold, beta = 2)}
        with open(result_folder+'result.pickle', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'],
                                         output_dict=True)
    def load_model(self, folder="models/AnoBeat/"):
        self.netE.load_state_dict(torch.load(folder+"encoder.mod", map_location=self.device))
        self.netD.load_state_dict(torch.load(folder+"decoder.mod", map_location=self.device))
        self.netDis_latent.load_state_dict(torch.load(folder+"discriminator_latent.mod", map_location=self.device))
        self.netDis_visual.load_state_dict(torch.load(folder+"discriminator_visual.mod", map_location=self.device))
        # Load the threeshold
        self.threeshold = np.load(folder+"thr.npy")
        self.trained = True
    

class FAnoGan:
    #ENCODER
    
    class Encoder(nn.Module):
        def __init__(self, ngpu, L, N, nef, nz):
            super(FAnoGan.Encoder, self).__init__()
            self.ngpu = ngpu
            self.conv = nn.Sequential(
                # input Beat dimensions L x N => L x 256 
                nn.Conv1d( L, nef, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef),
                nn.LeakyReLU(True),
                # state size (nef) x 128 
                nn.Conv1d( nef, nef * 2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 2),
                nn.LeakyReLU(True),
                # state size. (nef*2) x 64
                nn.Conv1d(nef * 2, nef * 4, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 4),
                nn.LeakyReLU(True),
                # state size. (nef*4) x 32
                nn.Conv1d( nef * 4, nef * 8, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 8),
                nn.LeakyReLU(True),
                # state size. (nef*8) x 16
                nn.Conv1d( nef * 8, nef * 16, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 16),
                nn.LeakyReLU(True),
                # state size. (nef * 16) x 8
                nn.Conv1d( nef * 16, nz, 8, 1, 0, bias=False),
                nn.LeakyReLU(True),
                # state size. (nef * 32) x 1
                nn.Flatten(),
                nn.Tanh())
        
        def forward(self, x):
            return self.conv(x)
    
    #DECODER
    
    class Decoder(nn.Module):
        def __init__(self, ngpu, nef, nz, nc, L):
            super(FAnoGan.Decoder, self).__init__()
            self.ngpu = ngpu
            self.upconv = nn.Sequential(
                
                nn.ConvTranspose1d( nz, nef * 16, 8, 1, 0, bias=False),
                nn.BatchNorm1d(nef * 16),
                nn.ReLU(True),
                # state size (nef*16) x 8 
                nn.ConvTranspose1d( nef*16, nef * 8, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 8),
                nn.ReLU(True),
                # state size. (nef*8) x 16 
                nn.ConvTranspose1d(nef * 8, nef * 4, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 4),
                nn.ReLU(True),
                # state size. (nef*4) x 32 
                nn.ConvTranspose1d( nef * 4, nef * 2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef * 2),
                nn.ReLU(True),
                # state size. (nef*2) x 64 
                nn.ConvTranspose1d( nef * 2, nef, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef),
                nn.ReLU(True),
                # state size. (nef) x 128 
                nn.ConvTranspose1d( nef, L, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (L) x 256 
            )
    
        def forward(self, z):
            out = self.upconv(z.unsqueeze(2))
            return out
    
    
    # DISCRIMINATOR
    
    class Discriminator_visible(nn.Module):
        def __init__(self, ngpu, L, N, nef, nz, nc, device = torch.device('cpu')):
            super(FAnoGan.Discriminator_visible, self).__init__()
            self.ngpu = ngpu
            self.device = device
            
            self.conv = nn.Sequential(
                # input Beat dimensions L x N => L x 256 
                nn.Conv1d( L, nef, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                # state size (nef) x 128 
                nn.Conv1d(nef, nef * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                # state size. (nef*2) x 64
                nn.Conv1d(nef * 2, nef * 4, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                # state size. (nef*4) x 32
                nn.Conv1d( nef * 4, nef * 8, 3, 2, 0, bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                # state size. (nef*8) x 16
                nn.Conv1d( nef * 8, nef * 16, 3, 2, 0, bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                # state size. (nef * 16) x 8
                )
            
            # Output layer
            self.output = nn.Sequential(
                nn.Conv1d( nef * 16, 1, 8, 1, 0, bias=False),
                nn.LeakyReLU(0.2),
                nn.Flatten())
            
        def features(self, x):
            return self.conv(x)
        
        def forward(self, x):
            return self.output(self.features(x))
    
    def __init__(self, device = torch.device('cpu'), N=280, L=1, nz=4,nc=3,nef=32, ngpu = 0):
        # Lenght of a beat.
        self.N = N
        
        # Number of leads
        self.L = L
        
        # Size of z latent vector 
        self.nz = nz
        
        # Size of additional information vector, one hot encoded sex value
        self.nc = nc # Example: [0, 1, 0] => [male, female, unlabeled] 
        
        # Size of feature maps 
        self.nef = nef

        
        self.device = device
        
        self.netE = self.Encoder(ngpu, L, N,nef, nz) 
        self.netD = self.Decoder(ngpu, nef, nz, nc, L)
        self.netDis_visible = self.Discriminator_visible(ngpu, L, N, nef, nz, nc, device = self.device)
        
        self.netE.to(device)
        self.netD.to(device)
        self.netDis_visible.to(device)
        
        # Initialize
        self.initialize()
        
        self.trained = False
        
        # Initialize the threeshold
        self.threeshold = 0
        
        # Covariance
        self.invSigma = np.diag(np.ones(self.N*self.L))
        
    
    def sample_noise(self, b_s):
        # Uniform Distribution [-1, 1]
        a = -1
        b = 1
        return (a - b) * torch.rand((b_s, self.nz), device=self.device, dtype=torch.float) + b
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    def initialize(self):
        self.netE.apply(self.weights_init)
        self.netD.apply(self.weights_init)
        self.netDis_visible.apply(self.weights_init)
        
    def get_anomaly_score(self, X, lambda_ano = 0.1):
        b_s = X.size(0) # Batch size
        X_rec = self.netD(self.netE(X)).detach()
        X = X.detach()
        residual_loss = torch.sum(torch.square_(X.view(b_s,-1) - X_rec.view(b_s,-1)), dim = 1).cpu().numpy()
        disc_loss = torch.sum(torch.abs(self.netDis_visible.features(X).view(b_s, -1)-
                                        self.netDis_visible.features(X_rec).view(b_s, -1)), dim = 1).detach().cpu().numpy()
        AS = (1.0-lambda_ano)*residual_loss + (lambda_ano)*disc_loss    
        return AS
    
    def get_rec_errors(self, X):
        b_s = X.size(0) # Batch size
        X_rec = self.netD(self.netE(X)).view(b_s,-1).detach().cpu().numpy()
        X = X.view(b_s,-1).detach().cpu().numpy() # (b_s, NxL)
        rec_errors = np.square(X_rec-X)
        return rec_errors
    
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1), device = self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.netDis_visible(interpolates)
        fake = torch.ones((real_samples.size(0), 1), device = self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def save(self, folder = '/'):
        torch.save(self.netDis_visible.state_dict(), folder+"discriminator_visible.mod")
        torch.save(self.netE.state_dict(), folder+"encoder.mod")
        torch.save(self.netD.state_dict(), folder+"decoder.mod")
        np.save( folder+"thr.npy", self.threeshold)
        
        
    def train(self, train_dl, valid_dl, num_epochs_gan, num_epochs_enc=20, lr = 0.0001, lr_disc = 0.0001, beta1=0.5, beta2 = 0.9, lambda_rec = 10,
              lambda_adv = 1, lambda_ejac = 0.1, lambda_dgp= 10, lambda_djac = 0.1, lambda_tv = 0.00001,
               n_critic = 5,model_folder= 'models/f_anoGAN/'):
        
     
        num_epochs = num_epochs_gan + num_epochs_enc
        beat_list = []
        MSE_losses = []
        ADV_losses = []
        Disc_visible_losses = []
        Disc_real_losses = []
        Disc_fake_losses = []
        
        ## Initialize Loss functions
        MSE = nn.MSELoss().to(self.device)
        
        # Create batch of latent vectors and labels that we will use to visualize

        fixed_noise = self.sample_noise(8)
        
        # Setup Adam optimizers for both Enc, Dec and Disc 
        optimizerE = optim.Adam(self.netE.parameters(), lr=lr, betas = (beta1, beta2))
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas = (beta1, beta2))
        optimizerDis_visible = optim.Adam(self.netDis_visible.parameters(), lr= lr_disc, betas=(beta1, beta2))
        
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        best_metric = 0
        
        print("Starting Training Loop...")
        print("\n \t ----------- TRAIN WGAN")
        for epoch in range(num_epochs_gan):
            # For each batch in the dataloader
            try:
                for i, data in enumerate(train_dl, 0):
                    # Format batch
                    X = data[0].to(self.device)
                    b_size = X.size(0)
                    X.requires_grad_(True)
                    
                    if i%n_critic == 0:
                        ##############################
                        # (1) Update D:
                        ##############################
                        self.netD.zero_grad()
    
                        Z_sampled = self.sample_noise(b_size)
                        X_sampled = self.netD(Z_sampled)
                        # adv loss
                        Disc_fake = self.netDis_visible(X_sampled).view(-1)
                        D_adv = -1 * lambda_adv * torch.mean(Disc_fake)
                        D_adv.backward()
                        optimizerD.step()
                        
                    ############################
                    # (3) Update Discriminator
                    ###########################
                    self.netDis_visible.zero_grad()
                    
                    # sampled
                    X_sampled = X_sampled.detach().requires_grad_(True)
                    Disc_real = self.netDis_visible(X).view(-1)
                    Disc_real = torch.mean(Disc_real)
                    Disc_fake = self.netDis_visible(X_sampled).view(-1)
                    Disc_fake = torch.mean(Disc_fake)
                    gp_sampled = self.compute_gradient_penalty(X, X_sampled)
                    Disc_visible_loss = -1 * Disc_real + Disc_fake + lambda_dgp * gp_sampled
                    Disc_visible_loss.backward()
                    
                    # Step
                    optimizerDis_visible.step()
                    
                    
                    ## Output training stats
                    if i % 50 == 0:
                        print('[%d/%d][%d/%d]\n\t Loss Disc visible: %.4f real: %.4f fake: %.4f '
                              % (epoch, num_epochs, i, len(train_dl),
                                 Disc_visible_loss.item(), Disc_real.item() ,Disc_fake.item()))
            
                    # Save Losses for plotting later
                    Disc_visible_losses.append(Disc_visible_loss.item())
                    Disc_real_losses.append(Disc_real.item())
                    Disc_fake_losses.append(Disc_fake.item())
                
            except KeyboardInterrupt:
                print("\n Interrupted")
                break
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if epoch % 20 == 0:
                with torch.no_grad():
                    fake = self.netD(fixed_noise).detach().cpu()
                beat_list.append(fake)
            
        print("\n \t -------------TRAIN ENCODER")
        for epoch in range(num_epochs_enc):
            try:
                for i, data in enumerate(train_dl, 0):
                    # Format batch
                    X = data[0].to(self.device)
                    b_size = X.size(0)
                    # Train Encoder to reduce MSE and fool discriminator
                    self.netE.zero_grad()
                    Z = self.netE(X)
                    X_rec = self.netD(Z)
                    E_mse = lambda_rec * MSE(X, X_rec)
                    E_adv = lambda_adv * MSE(self.netDis_visible.features(X),
                                                                         self.netDis_visible.features(X_rec))
                    loss = E_mse + E_adv 
                    loss.backward()
                    optimizerE.step()
                    ## Output training stats
                    if i % 50 == 0:
                        print('[%d/%d][%d/%d]\n\t mse Loss: %.4f \n\t adv loss: %.4f'
                              % (epoch, num_epochs, i, len(train_dl),
                                 E_mse.item(), E_adv.item()))
                    MSE_losses.append(E_mse.item())
                    ADV_losses.append(E_adv.item())
            except KeyboardInterrupt:
                print("interrupted")
                break
            # evaluate and save best model
            print("\n Evaluate")
            self.compute_threeshold(valid_dl, optimal=True)
            labels, pred = self.evaluate(valid_dl)
            metric = average_precision_score(labels, pred)
            if metric >= best_metric:
                print("\n %.4f -----> %.4f "%(best_metric, metric))
                best_metric = metric
                self.save(model_folder)
            else:
                print("\n %.4f less than best %.4f"%(metric, best_metric))
            
            
        self.load_model(model_folder)
    
        f, ax = plt.subplots(2, 1, sharex = True)
        ax[0].set_title("Adversarial Losses ")
        ax[0].plot(MSE_losses, label='mse error')
        ax[0].plot(ADV_losses, label = "adv loss")
        ax[0].legend()
        ax[1].set_title("Discriminator losses")
        ax[1].plot(Disc_visible_losses, label = 'visible rec')
        ax[1].plot(Disc_real_losses, label = 'real')
        ax[1].plot(Disc_fake_losses, label = 'fake')
        ax[1].legend()
        ax[1].set_ylabel("Loss")
        plt.xlabel("iterations")
        plt.show()
        for i in range(len(beat_list)):
            plot_some_beat(beat_list[i], suptitle = 'epoch: '+str(i))
        self.trained = True
        
    def evaluate(self, dl, verbose = False):
        pred = None
        labels = None
        if verbose:
            pbar = tqdm(total=len(dl))
        for i,data in enumerate(dl, 0):
            beats = data[0].to(self.device)
            label = np.array(data[2]['label']) == 'abnormal' # 0 normale 1 anormale
            ano_sc = self.get_anomaly_score(beats)
            if pred is None:
                pred = ano_sc
                labels = label
            else:
                pred = np.concatenate((pred, ano_sc), axis = 0)
                labels = np.concatenate((labels, label), axis = 0)
            if verbose:
                pbar.update(n=1)
        if verbose:
            pbar.close()
        return labels, pred
    
    def compute_threeshold(self, dl, optimal = False, verbose = False):
        labels, pred = self.evaluate(dl, verbose = verbose)
        # Compute threeshold using both errors
        if optimal:
            quantile_thr, best_thr = get_threeshold(pred, labels = labels, verbose = verbose)
            self.threeshold = best_thr
        else:
            quantile_thr, _ = get_threeshold(pred, labels = None, verbose = verbose)
            self.threeshold = quantile_thr
            
    def test(self, test_dl, result_folder='models/f_anoGAN'):
        if not self.trained:
            print("train or load a model")
            return None
        print("\n Starting Testing Loop... \n")
        
        labels,pred = self.evaluate(test_dl, verbose = True)
        
        clrp = classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'])
        print('\n')
        print(clrp)
        plot_error_mixing(pred, labels, self.threeshold)
        rep = classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'],
                                         output_dict=True)
        auc = roc_auc_score(labels, pred)
        pr_auc = average_precision_score(labels, pred)
        result = {'rep':rep['abnormal'], 'roc_auc':auc, 'pr_auc':pr_auc,
                  'pr_curve':precision_recall_curve( labels, pred),
                  'f2score': fbeta_score(labels, pred >= self.threeshold, beta = 2)}
        with open(result_folder+'result.pickle', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        plot_error_mixing(pred, labels, self.threeshold)
        return classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'],
                                         output_dict=True)
    def load_model(self, folder):
        self.netE.load_state_dict(torch.load(folder+"encoder.mod", map_location=self.device))
        self.netD.load_state_dict(torch.load(folder+"decoder.mod", map_location=self.device))
        self.netDis_visible.load_state_dict(torch.load(folder+"discriminator_visible.mod", map_location = self.device))
        # Load the threeshold
        self.threeshold = np.load(folder+"thr.npy")
        self.trained = True
    
    
        
        
