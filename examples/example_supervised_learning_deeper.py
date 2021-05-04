import numpy as np
import GPy
from pylab import *
from sys import path
np.random.seed(42)
import deepgp
# Utility to load sample data. It can be installed with pip. Otherwise just load some other data to run the demo.
import pods

#--------------------  DATA PREPARATION ---------------#
# Load some mocap data.
data = pods.datasets.cmu_mocap_35_walk_jog()

Ntr = 400 # Number of training points to use 
Nts = 500 # Number of test points to use

# All data represented in data['Y'], which is the angles of the movement of the subject
perm = np.random.permutation(Ntr+Nts) # Random selection of data to form train/test set
index_training = np.sort(perm[0:Ntr])
index_test     = np.sort(perm[Ntr:Ntr+Nts])
data_all_tr = data['Y'][index_training,:]
data_all_ts = data['Y'][index_test,    :]


# Some of the features (body joints) to be used as inputs, and some as outputs
X_tr = data_all_tr[:,0:55].copy() # Training inputs
Y_tr = data_all_tr[:, 55:].copy() # Training outputs
X_ts = data_all_ts[:,0:55].copy() # Test inputs 
Y_ts = data_all_ts[:, 55:].copy() # Test outputs

# It can help to normalize the input and/or output data.
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
scalerY = StandardScaler()
scalerX.fit(X_tr)
scalerY.fit(Y_tr)
X_tr_scaled = scalerX.transform(X_tr)
X_ts_scaled = scalerX.transform(X_ts)
Y_tr_scaled = scalerY.transform(Y_tr)


#--------- Model Construction ----------#
# We contruct the model counting layers from the output layer going "upwards":
#
# X ------> H2 -----> H1 ------> Y
#     f3        f2         f1
#  (kern3)    (kern2)    (kern1)


# Number of latent dimensions (single hidden layer, since the top layer is observed)
QX = X_tr.shape[1] # This *has* to be set equal to the dimension of X_tr since X_tr is observed
Q1 = 4 # Dimensionality for H1 
Q2 = 5 # Dimensionality for H2

# Define what kernels to use per layer. 
kern1 = GPy.kern.RBF(Q1,ARD=True) + GPy.kern.Bias(Q1)  # Layer H1 -> Y 
kern2 = GPy.kern.RBF(Q2,ARD=True) + GPy.kern.Bias(Q2)  # Layer H2 -> H1 
kern3 = GPy.kern.RBF(QX,ARD=False) + GPy.kern.Bias(QX) # Layer X  -> H2

# Number of inducing points to use
num_inducing = 40
# Whether to use back-constraint for variational posterior
back_constraint = False
# Dimensions of the MLP back-constraint if set to true
encoder_dims=[[300],[150], [150]]

# Collect the dimensions of [Y, H1, H2, X]
dimensions = [Y_tr.shape[1], Q1, Q2, QX] 

# Create model
m = deepgp.DeepGP(dimensions, Y_tr_scaled, X=X_tr_scaled,kernels=[kern1, kern2, kern3], num_inducing=num_inducing, back_constraint=back_constraint)

# print(m) # This will print the model

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
m_GP = GPy.models.SparseGPRegression(X=X_tr_scaled, Y=Y_tr_scaled, kernel=GPy.kern.RBF(QX)+GPy.kern.Bias(QX), num_inducing=num_inducing)
m_GP.Gaussian_noise.variance = m_GP.Y.var()*0.01
m_GP.Gaussian_noise.variance.fix()
m_GP.optimize(max_iters=100, messages=True)
m_GP.Gaussian_noise.variance.unfix()
m_GP.optimize(max_iters=400, messages=True)

def rmse(predictions, targets):
    return np.sqrt(((predictions.flatten() - targets.flatten()) ** 2).mean())

# Pass the predictions through inverse scaling transform to compare them in the original data space
Y_pred    = scalerY.inverse_transform(m.predict(X_ts_scaled)[0])
Y_pred_s  = scalerY.inverse_transform(m.predict_withSamples(X_ts_scaled, nSamples=500)[0])
Y_pred_GP = scalerY.inverse_transform(m_GP.predict(X_ts_scaled)[0])

print('# RMSE DGP               : ' + str(rmse(Y_pred, Y_ts)))
print('# RMSE DGP (with samples): ' + str(rmse(Y_pred_s, Y_ts)))
print('# RMSE GP                : ' + str(rmse(Y_pred_GP, Y_ts)))