# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import GPy
from GPy.util.pca import PCA
from GPy.core.parameterization.variational import VariationalPosterior, NormalPosterior
import sys

def initialize_latent(init, datanum, input_dim, Y):
    Xr = np.random.randn(datanum, input_dim)
    if init == 'PCA':
        # print 'Initializing latent with PCA...'
        p = PCA(Y)
        PC = p.project(Y, min(input_dim, Y.shape[1]))
        Xr[:PC.shape[0], :PC.shape[1]] = PC
        var = .1*p.fracs[:input_dim]
    elif init == 'bgplvm':
        # print 'Initializing latent with bgplvm...'
        # m = GPy.models.BayesianGPLVM(Y, input_dim, kernel=kernel, num_inducing=20, init_x='PCA')
        X_var = 0.5*np.ones((datanum, input_dim)) + 0.05*np.random.randn(datanum, input_dim)
        likelihood = GPy.likelihoods.Gaussian(variance = Y.var()*0.01)

        m = GPy.models.BayesianGPLVM(Y, input_dim, likelihood=likelihood, init='PCA', num_inducing=np.min((Y.shape[0], 25)), X_variance = X_var)
        m['Gaussian_noise.variance'].fix()
        m.optimize(max_iters=300,messages=False)
        m['Gaussian_noise.variance'].constrain_positive()
        m.optimize(max_iters=50,messages=False)
        Xr = m.X.mean
        var = X_var
        #print m
        print('Init SNR:' + str(Y.var() / m['Gaussian_noise.variance']))
    elif init == 'randomProjection':
        # print 'Initializing latent with Random projection...'

        Ycent = (Y-Y.mean())/Y.std()
        rr = np.random.rand(Ycent.shape[1], input_dim)
        Xr = np.dot(Ycent,rr)
        var = Xr.var(0)
    else:
        # print 'Initializing latent with Random...'
        var = Xr.var(0)

    if init not in ['bgplvm','randomProjection']:
        Xr -= Xr.mean(0)
        Xr /= Xr.std(0)

    return Xr, var/var.max()

def check_snr(m, messages=True):
    snr = []
    for i in range(len(m.layers)):
        if hasattr(m.layers[i],'views'):
            snr.append(list())
            for j in range(len(m.layers[i].views)):
                if isinstance(m.layers[i].views[j].Y, NormalPosterior) or isinstance(m.layers[i].views[j].Y, VariationalPosterior):
                    cur_var = m.layers[i].views[j].Y.mean.var()
                else:
                    cur_var = m.layers[i].views[j].Y.var()
                cur_snr = cur_var / m.layers[i].views[j].Gaussian_noise.variance.values
                if messages:
                    print('SNR layer ' + str(i) + ' view ' + str(j) + ':' + str(cur_snr))
                snr[-1].append(cur_snr)
        else:
            if isinstance(m.layers[i].Y, NormalPosterior) or isinstance(m.layers[i].Y, VariationalPosterior):
                cur_var = m.layers[i].Y.mean.var()
            else:
                cur_var = m.layers[i].Y.var()
            cur_snr = cur_var / m.layers[i].Gaussian_noise.variance.values
            if messages:
                print('SNR layer ' + str(i) + ':' + str(cur_snr))
            snr.append(cur_snr)
        sys.stdout.flush()
        return snr


def linsp(startP, endP):
    return np.linspace(startP, endP, endP - startP + 1)

def load_mocap_data(subjectsNum, motionsNum, standardise=True):
    # Download data (if they are not there already)
    #data_dir = '../../../GPy/GPy/util/datasets/mocap/cmu'

    #GPy.util.mocap.fetch_data(skel_store_dir=data_dir, motion_store_dir=data_dir,subj_motions=(subjectsNum, motionsNum), store_motions=True, return_motions=False)

    # Convert numbers to strings
    subjects = []
    motions = [list() for _ in range(len(subjectsNum))]
    for i in range(len(subjectsNum)):
        curSubj = str(int(subjectsNum[i]))
        if subjectsNum[i] < 10:
            curSubj = '0' + curSubj
        subjects.append(curSubj)
        for j in range(len(motionsNum[i])):
            curMot = str(int(motionsNum[i][j]))
            if motionsNum[i][j] < 10:
                curMot = '0' + curMot
            motions[i].append(curMot)


    Y = np.zeros((0,62))
    for i in range(len(subjects)):
        data = GPy.util.datasets.cmu_mocap(subjects[i], motions[i])
        Y = np.concatenate((Y,data['Y']))

    # Make figure move in place.
    # Y[:, 0:3] = 0.0
    Y = Y[:, 3:]

    meanY = Y.mean(axis=0)
    Ycentered = Y - meanY
    stdY = Ycentered.std(axis=0)
    stdY[np.where(stdY == 0)] = 1
    # Standardise
    if standardise:
        Y = Ycentered
        Y /= stdY
    return (Y, meanY, stdY)

def transform_labels(l):
    import numpy as np
    
    if l.shape[1] == 1:
        l_unique = np.unique(l)
#         K = len(l_unique)
        ret = np.zeros((l.shape[0],np.max(l_unique)+1))
        for i in l_unique:
            ret[np.where(l==i)[0],i] = 1
    else:
        ret = np.argmax(l,axis=1)[:,None]
    return ret

def visualize_DGP(model, labels, layer=0, dims=[0,1]):
    """
    A small utility to visualize the latent space of a DGP.
    """
    import matplotlib.pyplot as plt

    colors = ['r','g', 'b', 'm']
    markers = ['x','o','+', 'v']
    for i in range(model.layers[layer].X.mean.shape[0]):
        plt.scatter(model.layers[layer].X.mean[i,0],model.layers[layer].X.mean[i,1],color=colors[labels[i]], s=16, marker=markers[labels[i]])
