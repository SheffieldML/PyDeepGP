import numpy as np
import GPy
from pylab import *
from sys import path
np.random.seed(42)

import deepgp

# Utility to load sample data. It can be installed with pip. Otherwise just load some other data.
import pods

#--------------------  DATA PREPARATION ---------------#
# Load some mocap data.
data = pods.datasets.cmu_mocap_35_walk_jog()

Ntr = 100
Nts = 500

# All data represented in Y_all, which is the angles of the movement of the subject
Y_all = data['Y']
perm = np.random.permutation(Ntr+Nts)
index_training = np.sort(perm[0:Ntr])
index_test     = np.sort(perm[Ntr:Ntr+Nts])

Y_all_tr = Y_all[index_training,:]
Y_all_ts = Y_all[index_test,    :]


# Some of the features (body joints) to be used as inputs, and some as outputs
X_tr = Y_all_tr[:,0:55].copy()
Y_tr = Y_all_tr[:, 55:].copy()

X_ts = Y_all_ts[:,0:55].copy()
Y_ts = Y_all_ts[:, 55:].copy()

# TODO: You might need to normalize the input and/or output data.


#--------- Model Construction ----------#

# Number of latent dimensions (single hidden layer, since the top layer is observed)
Q = 5
# Define what kernels to use per layer
kern1 = GPy.kern.RBF(Q,ARD=True) + GPy.kern.Bias(Q)
kern2 = GPy.kern.RBF(Q,ARD=False) + GPy.kern.Bias(X_tr.shape[1])
# Number of inducing points to use
num_inducing = 40
# Whether to use back-constraint for variational posterior
back_constraint = False
# Dimensions of the MLP back-constraint if set to true
encoder_dims=[[300],[150]]

m = deepgp.DeepGP([Y_tr.shape[1],Q,X_tr.shape[1]],Y_tr, X=X_tr,kernels=[kern1, kern2], num_inducing=num_inducing, back_constraint=back_constraint)



#--------- Optimization ----------#
# Make sure initial noise variance gives a reasonable signal to noise ratio.
# Fix to that value for a few iterations to avoid early local minima
for i in range(len(m.layers)):
    output_var = m.layers[i].Y.var() if i==0 else m.layers[i].Y.mean.var()
    m.layers[i].Gaussian_noise.variance = output_var*0.01
    m.layers[i].Gaussian_noise.variance.fix()

m.optimize(max_iters=800, messages=True)
# Unfix noise variance now that we have initialized the model
for i in range(len(m.layers)):
    m.layers[i].Gaussian_noise.variance.unfix()

m.optimize(max_iters=1500, messages=True)

#--------- Inspection ----------#
# Compare with GP
m_GP = GPy.models.SparseGPRegression(X=X_tr, Y=Y_tr, kernel=GPy.kern.RBF(X_tr.shape[1])+GPy.kern.Bias(X_tr.shape[1]), num_inducing=num_inducing)
m_GP.Gaussian_noise.variance = m_GP.Y.var()*0.01
m_GP.Gaussian_noise.variance.fix()
m_GP.optimize(max_iters=100, messages=True)
m_GP.Gaussian_noise.variance.unfix()
m_GP.optimize(max_iters=400, messages=True)

def rmse(predictions, targets):
    return np.sqrt(((predictions.flatten() - targets.flatten()) ** 2).mean())

Y_pred = m.predict(X_ts)[0]
Y_pred_s = m.predict_withSamples(X_ts, nSamples=500)[0]
Y_pred_GP = m_GP.predict(X_ts)[0]

print('# RMSE DGP               : ' + str(rmse(Y_pred, Y_ts)))
print('# RMSE DGP (with samples): ' + str(rmse(Y_pred_s, Y_ts)))
print('# RMSE GP                : ' + str(rmse(Y_pred_GP, Y_ts)))
