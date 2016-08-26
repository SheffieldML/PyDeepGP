# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np

class BernoulliEntropy(object):
    
    def comp_value(self, qS):
        return (qS*np.log(qS)).sum() + ((1-qS)*np.log(1-qS)).sum()

    def update_gradients(self, qS):
        qS.gradient += np.log((1-qS)/qS)

class NormalEntropy(object):
    constant = 1.+np.log(2.*np.pi)
    def __init__(self, scale=1.):
        self.scale = scale

    def comp_value(self, variational_posterior):
        var = variational_posterior.variance
        return self.scale*(-(NormalEntropy.constant+np.log(var)).sum()/2.)

    def update_gradients(self, variational_posterior):
        variational_posterior.variance.gradient +=  self.scale*(1. / (variational_posterior.variance*2.))
        
class NormalPrior(object):
    
    def comp_value(self, variational_posterior):
        var_mean = np.square(variational_posterior.mean).sum()
        var_S = (variational_posterior.variance - np.log(variational_posterior.variance)).sum()
        return 0.5 * (var_mean + var_S) - 0.5 * variational_posterior.input_dim * variational_posterior.num_data

    def update_gradients(self, variational_posterior):
        # dL:
        variational_posterior.mean.gradient -= variational_posterior.mean
        variational_posterior.variance.gradient -= (1. - (1. / (variational_posterior.variance)))/2.
        