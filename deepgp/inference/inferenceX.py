
import numpy as np
import GPy
from GPy.core import Model
from GPy.core.parameterization import variational
from GPy.util.linalg import tdot, jitchol, dtrtrs, dtrtri

class InferenceX(Model):
    """
    The model class for inference of new X with given new Y. (replacing the "do_test_latent" in Bayesian GPLVM)
    It is a tiny inference model created from the original GP model. The kernel, likelihood (only Gaussian is supported at the moment) 
    and posterior distribution are taken from the original model.
    For Regression models and GPLVM, a point estimate of the latent variable X will be inferred. 
    For Bayesian GPLVM, the variational posterior of X will be inferred. 
    X is inferred through a gradient optimization of the inference model.

    :param model: the GPy model used in inference
    :type model: GPy.core.Model
    :param Y: the new observed data for inference
    :type Y: numpy.ndarray
    :param init: the distance metric of Y for initializing X with the nearest neighbour.
    :type init: 'L2', 'NCC' and 'rand'
    """
    def __init__(self, model, Y, name='inferenceX', init='L2'):
        if np.isnan(Y).any() or getattr(model, 'missing_data', False):
            assert Y.shape[0]==1, "The current implementation of inference X only support one data point at a time with missing data!"
            self.missing_data = True
            self.valid_dim = np.logical_not(np.isnan(Y[0]))
            self.ninan = getattr(model, 'ninan', None)
        else:
            self.missing_data = False
        super(InferenceX, self).__init__(name)
        self.likelihood = model.likelihood.copy()
        self.kern = model.kern.copy()

        from copy import deepcopy
        self.posterior = deepcopy(model.posterior)
        self.uncertain_input = False
#         if isinstance(model.X, variational.VariationalPosterior):
#             self.uncertain_input = True
#         else:
#             self.uncertain_input = False
        if hasattr(model, 'Z'):
            self.sparse_gp = True
            self.Z = model.Z.copy()
        else:
            self.sparse_gp = False
            self.uncertain_input = False
            self.Z = model.X.copy()
        self.Y = Y
        self.X = self._init_X(model, Y, init=init)
        self.compute_dL()

        self.link_parameter(self.X)

    def _init_X(self, model, Y_new, init='L2'):
        # Initialize the new X by finding the nearest point in Y space.

        Y = model.Y
        if self.missing_data:
            Y = Y[:,self.valid_dim]
            Y_new = Y_new[:,self.valid_dim]
            dist = -2.*Y_new.dot(Y.T) + np.square(Y_new).sum(axis=1)[:,None]+ np.square(Y).sum(axis=1)[None,:]
        else:
            if init=='L2':
                dist = -2.*Y_new.dot(Y.T) + np.square(Y_new).sum(axis=1)[:,None]+ np.square(Y).sum(axis=1)[None,:]
            elif init=='NCC':
                dist = Y_new.dot(Y.T)
            elif init=='rand':
                dist = np.random.rand(Y_new.shape[0],Y.shape[0])
        idx = dist.argmin(axis=1)

        from GPy.core import Param
        if  isinstance(model.X, variational.VariationalPosterior):
            X = Param('latent mean',model.X.mean.values[idx].copy())
            X.set_prior(GPy.core.parameterization.priors.Gaussian(0.,1.), warning=False)
        else:
            X = Param('latent mean',(model.X[idx].values).copy())
        return X

    def compute_dL(self):
        # Common computation
        beta = 1./np.fmax(self.likelihood.variance, 1e-6)
        output_dim = self.Y.shape[-1]
        wv = self.posterior.woodbury_vector
        if self.missing_data:
            wv = wv[:,self.valid_dim]
            output_dim = self.valid_dim.sum()
            if self.ninan is not None:
                self.dL_dpsi2 = beta/2.*(self.posterior.woodbury_inv[:,:,self.valid_dim] - tdot(wv)[:, :, None]).sum(-1)
            else:
                self.dL_dpsi2 = beta/2.*(output_dim*self.posterior.woodbury_inv - tdot(wv))
            self.dL_dpsi1 = beta*np.dot(self.Y[:,self.valid_dim], wv.T)
            self.dL_dpsi0 = - beta/2.* np.ones(self.Y.shape[0])
        else:
            self.dL_dpsi2 = beta*(output_dim*self.posterior.woodbury_inv - tdot(wv))/2. #np.einsum('md,od->mo',wv, wv)
            self.dL_dpsi1 = beta*np.dot(self.Y, wv.T)
            self.dL_dpsi0 = -beta/2.*output_dim* np.ones(self.Y.shape[0])

    def parameters_changed(self):
        N, D = self.Y.shape

        Kss = self.kern.K(self.X)
        Ksu = self.kern.K(self.X, self.Z)

        wv = self.posterior.woodbury_vector
        wi = self.posterior.woodbury_inv
        
        a = self.Y - Ksu.dot(wv)
        
        C = Kss  + np.eye(N)*self.likelihood.variance - Ksu.dot(wi).dot(Ksu.T)
        Lc = jitchol(C)
        LcInva = dtrtrs(Lc, a)[0]
        LcInv = dtrtri(Lc)
        CInva = dtrtrs(Lc, LcInva,trans=1)[0]

        self._log_marginal_likelihood = -N*D/2.*np.log(2*np.pi) - D*np.log(np.diag(Lc)).sum() - np.square(LcInva).sum()/2.

        dKsu = CInva.dot(wv.T)
        dKss = tdot(CInva)/2. -D* tdot(LcInv.T)/2.
        dKsu += -2. * dKss.dot(Ksu).dot(wi)
        
        X_grad = self.kern.gradients_X(dKss, self.X)
        X_grad += self.kern.gradients_X(dKsu, self.X, self.Z)
        self.X.gradient = X_grad      
        
        if self.uncertain_input:
            # Update Log-likelihood
            KL_div = self.variational_prior.KL_divergence(self.X)
            # update for the KL divergence
            self.variational_prior.update_gradients_KL(self.X)
            self._log_marginal_likelihood += -KL_div

    def log_likelihood(self):
        return self._log_marginal_likelihood


