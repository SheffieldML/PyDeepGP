import numpy as np
import GPy
from pylab import *
from sys import path
np.random.seed(42)

import deepgp

# Utility to load sample data. It can be installed with pip. Otherwise just load some other data.
import pods

# Load data
data = pods.datasets.oil_100()
Y = data['X']
labels =  data['Y'].argmax(axis=1)


#--------- Model Construction ----------#

# Number of latent dimensions per layer
Q1, Q2 = 5, 4
# Type of kernel per layer
kern1 = GPy.kern.RBF(Q1,ARD=True) + GPy.kern.Bias(Q1)
kern2 = GPy.kern.RBF(Q2,ARD=True) + GPy.kern.Bias(Q2)
# Number of inducing points per layer (can be set to different if given as list).
num_inducing = 40
# Whether to use back-constraint for variational posterior
back_constraint = False
# Dimensions of the MLP back-constraint if set to true
encoder_dims=[[300],[150]]

m = deepgp.DeepGP([Y.shape[1],Q1,Q2],Y,kernels=[kern1,kern2], num_inducing=num_inducing, back_constraint=back_constraint, encoder_dims = encoder_dims)

#--------- Optimization ----------#
# Make sure initial noise variance gives a reasonable signal to noise ratio
for i in range(len(m.layers)):
    output_var = m.layers[i].Y.var() if i==0 else m.layers[i].Y.mean.var()
    m.layers[i].Gaussian_noise.variance = output_var*0.01

m.optimize(max_iters=5000, messages=True)


#--------- Inspection ----------#
# Plot ARD scales per layer
m.obslayer.kern.plot_ARD()
m.layer_1.kern.plot_ARD()

# From the plots above, we see which ones are the dominant dimensions for each layer. 
# So we use these dimensions in the visualization of the latent space below.
plt.figure(figsize=(5,5))
deepgp.util.visualize_DGP(m, labels, layer=0, dims=[1,2]); plt.title('Layer 0')
plt.figure(figsize=(5,5))
deepgp.util.visualize_DGP(m, labels, layer=1, dims=[0,1]); plt.title('Layer 1')