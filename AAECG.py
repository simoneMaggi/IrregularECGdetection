#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 19:12:43 2021

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
from EarlyStopping import EarlyStopping
from tqdm import tqdm
import Utils as u
num_cores = multiprocessing.cpu_count()
import pickle



class CBeatAAE:
    #ENCODER
    class Encoder(nn.Module):
        def __init__(self, ngpu, L, N, nef, nz):
            super(CBeatAAE.Encoder, self).__init__()
            self.ngpu = ngpu
            self.nz = nz
            self.conv = nn.Sequential(
                # input Beat dimensions L x N => L x N 
                nn.Conv1d( L, nef, 4, 2, 1, bias=False),
                nn.LeakyReLU(True, inplace=True),
                # state size (nef) x 128 
                nn.Conv1d( nef, nef * 2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef*2),
                nn.LeakyReLU(True, inplace=True),
                # state size. (nef*2) x 64
                nn.Conv1d(nef * 2, nef * 4, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef*4),
                nn.LeakyReLU(True, inplace=True),
                # state size. (nef*4) x 32
                nn.Conv1d( nef * 4, nef * 8, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef*8),
                nn.LeakyReLU(True, inplace=True),
                # state size. (nef*8) x 16
                nn.Conv1d( nef * 8, nef * 16, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef*16),
                nn.LeakyReLU(True, inplace=True)
                # state size. (nef * 16) x 8
                )
            self.mu = nn.Sequential(
                nn.Conv1d( nef * 16, nz, 8, 1, 0 ),
                # state size. (nef * 32) x 1
                nn.Flatten()
                )
            self.logvar = nn.Sequential(
                nn.Conv1d( nef * 16, nz, 8, 1, 0),
                # state size. (nef * 32) x 1
                nn.Flatten()
                )
            
        
        def reparametrization(self, mu, logvar):
            std = torch.exp(logvar/2)
            sampled_z = torch.randn((mu.size(0), self.nz), device = mu.device)
            z = sampled_z * std + mu
            
            return z
        
        def features(self, x):
            return self.conv(x)
        
        def forward(self, x):
            f = self.conv(x)
            return self.reparametrization(self.mu(f), self.logvar(f))
        
        def variance(self, x):
            f = self.conv(x)
            std = torch.exp(self.logvar(f)/2)
            return std
    
    #DECODER
    
    class Decoder(nn.Module):
        def __init__(self, ngpu, nef, nz, nc, L):
            super(CBeatAAE.Decoder, self).__init__()
            self.ngpu = ngpu
            self.upconv = nn.Sequential(
                # state size (nef*16) x 8 
                nn.ConvTranspose1d( nz + nc, nef * 16, 8, 1, 0, bias=False),
                nn.BatchNorm1d(nef * 16),
                nn.ReLU(True),
                # state size. (nef*8) x 16 
                nn.ConvTranspose1d(nef * 16, nef * 8, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 8),
                nn.ReLU(True),
                # state size. (nef*4) x 32 
                nn.ConvTranspose1d( nef * 8, nef * 4, 3, 2, 0, bias=False),
                nn.BatchNorm1d(nef * 4),
                nn.ReLU(True),
                # state size. (nef*2) x 64 
                nn.ConvTranspose1d( nef * 4, nef * 2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef*2),
                nn.ReLU(True),
                # state size. 
                nn.ConvTranspose1d( nef * 2, nef, 4, 2, 1, bias=False),
                nn.BatchNorm1d(nef),
                nn.ReLU(True),
                # state size. (nef) x 128 
                nn.ConvTranspose1d( nef, L, 4, 2, 1, bias=False),
                # state size. (L) x 256 
                nn.Tanh()
            )
    
        def forward(self, z, labels):
            l = torch.cat((z,labels), dim = 1)
            return self.upconv(l.unsqueeze(2))
    
    
    # DISCRIMINATOR
    
    class Discriminator_prior(nn.Module):
        def __init__(self, ngpu, nz, ndf):
            super(CBeatAAE.Discriminator_prior, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                nn.Linear(in_features=(nz), out_features=ndf),
                nn.LeakyReLU(0.2),
                
                nn.Linear(in_features=ndf, out_features=ndf//2),
                nn.LeakyReLU(0.2),
                
                nn.Linear(in_features=ndf//2, out_features=ndf//4),
                nn.LeakyReLU(0.2),
                
                nn.Linear(in_features=ndf//4, out_features=1)
            )
    
        def forward(self, input):
            return self.main(input)
    
    
    def __init__(self, device = torch.device('cpu'), N=280, L=1, nz=3,nc=3,nef=32,ndf=32, ngpu = 0):
        # Lenght of a beat.
        self.N = N
        
        # Number of leads
        self.L = L
        
        # Size of z latent vector 
        self.nz = nz
        
        # Size of additional information vector, one hot encoded sex value
        self.nc = nc # Example: [0, 1, 0] => [male, female, unlabeled] 
        
        # Size of feature maps in encoder
        self.nef = nef
        
        # Size of feature maps in discriminator
        self.ndf = ndf
        
        self.device = device
        
        self.netE = self.Encoder(ngpu, L, N,nef, nz) 
        self.netD = self.Decoder(ngpu, nef, nz, nc, L)
        self.netDis_prior = self.Discriminator_prior(ngpu, nz, ndf)
        
        self.netE.to(device)
        self.netD.to(device)
        self.netDis_prior.to(device)
        
        # Initialize
        self.initialize()
        
        self.trained = False
        
        # Initialize the threeshold
        self.threeshold = 0
        
        # Covariance
        self.invSigma = np.diag(np.ones(self.N*self.L))
        
    def initialize(self):
        self.netE.apply(self.weights_init)
        self.netD.apply(self.weights_init)
        self.netDis_prior.apply(self.weights_init)
        
    def sample_noise(self, b_s, uniform_noise = False):
        if uniform_noise:
            # Uniform Distribution [-1, 1]
            a = -1
            b = 1
            noise = (a - b) * torch.rand((b_s, self.nz), device=self.device, dtype=torch.float) + b
        else:
            noise = torch.randn((b_s, self.nz), device = self.device, dtype = torch.float)
        return noise
    
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
    
    def plot_error_mixing(self, errors, labels, best_thr, est_thr = None,):
        plt.figure()
        h = np.histogram(errors[np.where(labels == 0)])
        plt.plot(h[1][:-1], h[0]/np.sum(h[0]), label = 'normal')
        h = np.histogram(errors[np.where(labels == 1)])
        plt.plot( h[1][:-1], h[0]/np.sum(h[0]), label = 'abnormal')
        plt.ylim([0, 1.001])
        plt.xlim([0, np.max(errors)])
        if est_thr is None:
            plt.vlines(x = best_thr, ymin = 0, ymax = 1, colors = 'b', label = 'threeshold')
        else:
            plt.vlines(x = est_thr, ymin = 0, ymax = 1, colors = 'b', label = 'alpha-thr')
            plt.vlines(x = best_thr, ymin = 0, ymax = 1, colors = 'r', label = 'best f1-thr')
        plt.legend()
        plt.show()
    
    def get_threeshold(self, errors, labels = None, alpha = 0.05, verbose = False):
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
        if labels is None:
            return thr
        quantile_thr = thr
        best_thr = 0
        # Estimate threeshold with both
        thrs = np.linspace(np.min(errors), np.max(errors), num=errors.shape[0]*2)
        if verbose:
            print("estimate threeshold with all data")
        e = Parallel(n_jobs=num_cores)(delayed(fbeta_score)( labels, errors >= t, beta = 2)
                               for t in thrs)
        ind = np.argmax(np.array(e))
        best_thr = thrs[ind]
        if verbose:
           self.plot_error_mixing(errors, labels, best_thr, est_thr = thr)
        
        return quantile_thr, best_thr
    
    
    def get_anomaly_score(self, X, label, L=1):
        # X is b_s x L x N
        b_s = X.size(0) # Batch size
        AS = 0
        X_det = X.view(b_s,-1).detach().cpu().numpy()
        for l in range(L):
            X_rec = self.netD(self.netE(X), label).view(b_s,-1).detach().cpu().numpy()
            AS += np.mean(np.square(X_rec - X_det), axis = 1)
        return AS/L
    
    def get_rec_errors(self, X, label):
        b_s = X.size(0) # Batch size
        X_rec = self.netD(self.netE(X), label).view(b_s,-1).detach().cpu().numpy()
        X = X.view(b_s,-1).detach().cpu().numpy() # (b_s, NxL)
        rec_errors = np.square(X_rec-X)
        return rec_errors

    
    def compute_gradient_penalty(self, real_samples, fake_samples, labels = None):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        if labels is None:
            alpha = torch.rand((real_samples.size(0), 1), device = self.device)
        else:
            alpha = torch.rand((real_samples.size(0), 1, 1), device = self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        if labels is None:
            d_interpolates = self.netDis_prior(interpolates)
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
    
    
    def train(self, train_dl, valid_dl, num_epochs, lr = 0.0001, beta1=0.5, beta2 = 0.9, lambda_rec = 0.999,
              lambda_adv = 0.001, lambda_gp = 10, lambda_tv = 0.001,
               n_critic = 5):                                                                             
        beat_list = []
        E_losses = []
        MSE_losses = []
        ADV_losses = []
        Disc_prior_losses = []
        Disc_visible_losses = []
        TV_losses = []
        
        ## Initialize Loss functions
        MSE = nn.MSELoss().to(self.device)
        
        # Create batch of latent vectors and labels that we will use to visualize
        #  the progression of the Decoder
        
        fixed_labels = torch.tensor(sorted([[1,0,0],[0,1,0]]*4), dtype = torch.float, 
                                    device = self.device)
        fixed_noise = self.sample_noise(8)
        
        # Establish convention for noise(positive) and latent(negative) code labels
        
        # Setup Adam optimizers for both Enc, Dec and Disc 
        optimizerE = optim.Adam(self.netE.parameters(), lr=lr, betas = (beta1, beta2))
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas = (beta1, beta2))
        optimizerDis_prior = optim.Adam(self.netDis_prior.parameters(), lr=lr, betas = (beta1, beta2))
        
        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            try:
                # For each batch in the dataloader
                for i, data in enumerate(train_dl, 0):
                    # Format batch
                    X = data[0].to(self.device)
                    label = data[1].to(self.device)
                    b_size = X.size(0)
                    ############################
                    # (2) Update Discriminator network on latent: minimize wasserstein distance
                    ###########################
                    for _ in range(n_critic):
                        self.netDis_prior.zero_grad()
                        data_n = next(iter(train_dl))
                        X_n = data_n[0].to(self.device)
                        Z_prior = self.sample_noise(b_size) 
                        Z_latent = self.netE(X_n).detach()
                        Disc_z_prior = self.netDis_prior(Z_prior).view(-1)
                        Disc_z_real = torch.mean(Disc_z_prior)
                        Disc_z = self.netDis_prior(Z_latent).view(-1)
                        Disc_z_fake = torch.mean(Disc_z)
                        gp = self.compute_gradient_penalty(Z_prior, Z_latent)
                        W_distance = Disc_z_real - Disc_z_fake 
                        loss = -1*W_distance + lambda_gp * gp
                        # Calculate gradients for Disc in backward pass
                        loss.backward()
                        optimizerDis_prior.step()
                    
                    ##############################
                    # (1) Update D and E networks:
                    ##############################
                    self.netE.zero_grad()
                    self.netD.zero_grad()
                    
                    Z = self.netE(X)
                    X_rec = self.netD(Z, label)
                    X_rec_detached = self.netD(Z.detach(), label)
                    
                    # total variation reg
                    TV_loss = torch.div(torch.abs(X_rec_detached[:,:,1:] - X_rec_detached[:,:,:-1]).sum(), b_size)
                    
                    # rec losses
                    ED_mse =  MSE(X, X_rec) 

                    # adv latent loss
                    E_advz =  torch.mean(self.netDis_prior(Z).view(-1))
                    
                    total_loss = lambda_rec * ED_mse + lambda_tv * TV_loss -1* lambda_adv *E_advz
                    total_loss.backward()
                    optimizerE.step()
                    optimizerD.step()
                    
                    ## Output training stats
                    if i % 50 == 0:
                        print('[%d/%d][%d/%d]\n\t mse Loss: %.4f \n\t  Adv loss Encoder: %.4f \n\t Loss Disc prior: %.4f  fake: %.4f real: %.4f \n\t  TV loss %.4f'
                              % (epoch, num_epochs, i, len(train_dl),
                                 ED_mse.item(), E_advz.item(), W_distance.item(),Disc_z_fake.item(), Disc_z_real.item(), TV_loss.item()))
            
                    # Save Losses for plotting later
                    E_losses.append(E_advz.item())
                    MSE_losses.append(ED_mse.item())
                    Disc_prior_losses.append(W_distance.item())
                    TV_losses.append(TV_loss.item())
            except KeyboardInterrupt:
                print("\n INTERRUPT DETECTED: training stopped")
                break
            if epoch % 20 == 0:
                with torch.no_grad():
                    fake = self.netD(fixed_noise, fixed_labels).detach().cpu()
                beat_list.append(fake)
        torch.save(self.netDis_prior.state_dict(), "models/AAE/discriminator_prior.mod")
        torch.save(self.netE.state_dict(), "models/AAE/encoder.mod")
        torch.save(self.netD.state_dict(), "models/AAE/decoder.mod")

        pred = None
        labels = None
        print("\t Computing the optimal threeshold")
        pbar = tqdm(total = len(valid_dl))
        print("\t Compute errors")
        for i,data in enumerate(valid_dl, 0):
            beats = data[0].to(self.device)
            codes = data[1].to(self.device)
            label = np.array(data[2]['label']) == 'abnormal' # 0 normale 1 anormale
            ano_sc = self.get_anomaly_score(beats, codes)
            if pred is None:
                pred = ano_sc
                labels = label
            else:
                pred = np.concatenate((pred, ano_sc), axis = 0)
                labels = np.concatenate((labels, label), axis = 0)
            pbar.update(n = 1)
        pbar.close()
        labels = np.array(labels)
        # Compute threeshold using both errors
        quantile_thr, best_thr = self.get_threeshold(pred, labels = labels, verbose = True)
        self.threeshold = best_thr
        # Save the best threeshold
        np.save("models/AAE/thr.npy", np.array([best_thr]))
        f, ax = plt.subplots(3, 1, sharex = True)
        ax[0].set_title("Adversarial Losses ")
        ax[0].plot(MSE_losses, label='mse error')
        ax[0].plot(ADV_losses, label = "adv loss")
        ax[0].plot(E_losses, label = 'Adv aggregate error')
        ax[0].legend()
        ax[1].set_title("Discriminator losses")
        ax[1].plot(Disc_prior_losses, label = 'latent')
        ax[1].plot(Disc_visible_losses, label = 'visible')
        ax[1].legend()
        ax[1].set_ylabel("Loss")
        ax[2].plot(TV_losses, label= "total variation")
        ax[2].legend()
        plt.xlabel("iterations")
        plt.show()
        for i in range(len(beat_list)):
            u.plot_some_beat(beat_list[i], suptitle = 'epoch: '+str(i), 
                             titles_list = [('M' if fixed_labels[j][0]==0 else 'F' ) for j in range(len(beat_list[i]))])
        self.trained = True
        
    def save(self, folder = '/'):
        torch.save(self.netDis_prior.state_dict(), folder+"discriminator_prior.mod")
        torch.save(self.netE.state_dict(), folder+"encoder.mod")
        torch.save(self.netD.state_dict(), folder+"decoder.mod")
        
    def load_ckpt(self, folder = '/'):
        self.netE.load_state_dict(torch.load(folder+"encoder.mod", map_location=self.device))
        self.netD.load_state_dict(torch.load(folder+"decoder.mod", map_location=self.device))
        self.netDis_prior.load_state_dict(torch.load(folder+"discriminator_prior.mod", map_location=self.device))
        
    def test(self, test_dl, result_folder = 'models/AAE/'):
        if not self.trained:
            print("train or load a model")
            return None
        print("\n Starting Testing Loop... \n")
        pred = None
        labels = None
        pbar = tqdm(total=len(test_dl))
        for i,data in enumerate(test_dl, 0):
            beats = data[0].to(self.device)
            codes = data[1].to(self.device)
            label = np.array(data[2]['label']) == 'abnormal' # 0 normale 1 anormale
            ano_sc = self.get_anomaly_score(beats, codes)
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
        self.plot_error_mixing(pred, labels, self.threeshold)
        rep = classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'],
                                         output_dict=True)
        auc = roc_auc_score(labels, pred)
        pr_auc = average_precision_score(labels, pred)
        result = {'rep':rep['abnormal'], 'roc_auc':auc, 'pr_auc':pr_auc,
                  'pr_curve':precision_recall_curve( labels, pred), 
                  'f2score': fbeta_score(label, pred >= self.threeshold, beta = 2)}
        
        with open(result_folder+'result.pickle', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return classification_report(labels, pred >= self.threeshold,
                                         target_names=['normal', 'abnormal'],
                                         output_dict=True)
    def load_model(self):
        self.netE.load_state_dict(torch.load("models/AAE/encoder.mod", map_location=self.device))
        self.netD.load_state_dict(torch.load("models/AAE/decoder.mod", map_location=self.device))
        self.netDis_prior.load_state_dict(torch.load("models/AAE/discriminator_prior.mod", map_location = self.device))
        # Load the threeshold
        self.threeshold = np.load("models/AAE/thr.npy")
        self.trained = True
    
    
        
print("ok")        