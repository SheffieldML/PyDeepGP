# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)




from GPy.util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri,pdinv
from GPy.util import diag
from GPy.core.parameterization.variational import VariationalPosterior
import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import Posterior

log_2_pi = np.log(2*np.pi)

try:
    from mpi4py import MPI
except:
    pass

class VarDTC(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    const_jitter = 1e-6
    def __init__(self, mpi_comm=None):

        self.mpi_comm = mpi_comm
        self.Y_speedup = False # Replace Y with the cholesky factor of YY.T, but the computation of posterior object will be skipped.
    
    def get_trYYT(self, Y):
        return np.sum(np.square(Y))

    def get_YYTfactor(self, Y):
        N, D = Y.shape
        if (N>=D):
            return Y.view(np.ndarray)
        else:
            return jitchol(tdot(Y))

    def gatherPsiStat(self, kern, X, Z, Y, beta, uncertain_inputs):

        assert beta.size == 1

        if uncertain_inputs:
            psi0 = kern.psi0(Z, X)
            psi1 = kern.psi1(Z, X)*beta
            psi2 = kern.psi2(Z, X)*beta
        else:
            psi0 = kern.Kdiag(X)
            psi1 = kern.K(X, Z)
            psi2 = np.dot(psi1.T,psi1)*beta
            psi1 = psi1*beta
            
        psi0 = psi0.sum()*beta
        
        if isinstance(Y, VariationalPosterior):
            m, s = Y.mean, Y.variance
            
            psi1Y = np.dot(m.T,psi1) # DxM
            Shalf = np.sqrt(s.sum(axis=1))
            psi1S = Shalf[:,None]*psi1
            
            YRY = (np.square(m).sum()+s.sum())*beta
        else:
            psi1Y = np.dot(Y.T,psi1) # DxM
            
            trYYT = self.get_trYYT(Y)
            YRY = trYYT*beta
            psi1S = None
            Shalf = None
            
        return psi0, psi2, YRY, psi1, psi1Y, Shalf, psi1S

    def inference(self, kern, X, Z, likelihood, Y, Y_metadata=None, Lm=None, dL_dKmm=None, Kuu_sigma=None):
        """
        The first phase of inference:
        Compute: log-likelihood, dL_dKmm

        Cached intermediate results: Kmm, KmmInv,
        """


        num_data, output_dim = Y.shape
        input_dim = Z.shape[0]

        uncertain_inputs = isinstance(X, VariationalPosterior)
        uncertain_outputs = isinstance(Y, VariationalPosterior)

        # from ..models.sslvm import Gaussian_Gamma
        # if isinstance(likelihood, Gaussian_Gamma):
        #     beta = likelihood.expectation_beta()
        #     logL_R = -num_data*likelihood.expectation_logbeta()
        # else:
        beta = 1./np.fmax(likelihood.variance, 1e-6)
        logL_R = -num_data*np.log(beta)


        psi0, psi2, YRY, psi1, psi1Y, Shalf, psi1S = self.gatherPsiStat(kern, X, Z, Y, beta, uncertain_inputs)

        #======================================================================
        # Compute Common Components
        #======================================================================

        Kmm = kern.K(Z).copy()
        if Kuu_sigma is not None:
            diag.add(Kmm, Kuu_sigma)
        else:
            diag.add(Kmm, self.const_jitter)
        Lm = jitchol(Kmm)

        #LmInv = dtrtri(Lm)
        if uncertain_inputs:
            LmInvPsi2LmInvT = backsub_both_sides(Lm, psi2, 'right')
        else:
            LmInvPsi2LmInvT = tdot(dtrtrs(Lm, psi1.T)[0])/beta #tdot(psi1.dot(LmInv.T).T) /beta
            
        Lambda = np.eye(Kmm.shape[0])+LmInvPsi2LmInvT
        LL = jitchol(Lambda)
        LmLL = Lm.dot(LL)
#        LLInv = dtrtri(LL)
 #       LmLLInv = LLInv.dot(LmInv)
        
        logdet_L = 2.*np.sum(np.log(np.diag(LL)))
        b  = dtrtrs(LmLL, psi1Y.T)[0].T #psi1Y.dot(LmLLInv.T)
        bbt = np.square(b).sum()
        v = dtrtrs(LmLL, b.T, trans=1)[0].T #b.dot(LmLLInv)
        LLinvPsi1TYYTPsi1LLinvT = tdot(b.T)
        
        if psi1S is not None:
            psi1SLLinv = dtrtrs(LmLL, psi1S.T)[0].T #psi1S.dot(LmLLInv.T)
            bbt += np.square(psi1SLLinv).sum()
            LLinvPsi1TYYTPsi1LLinvT += tdot(psi1SLLinv.T)
            psi1SP = dtrtrs(LmLL, psi1SLLinv.T, trans=1)[0].T #psi1SLLinv.dot(LmLLInv)
        tmp = -backsub_both_sides(LL, LLinvPsi1TYYTPsi1LLinvT+output_dim*np.eye(input_dim))
        dL_dpsi2R = backsub_both_sides(Lm, tmp+output_dim*np.eye(input_dim))/2
        #tmp = -LLInv.T.dot(LLinvPsi1TYYTPsi1LLinvT+output_dim*np.eye(input_dim)).dot(LLInv)
        #dL_dpsi2R = LmInv.T.dot(tmp+output_dim*np.eye(input_dim)).dot(LmInv)/2.
        
        #======================================================================
        # Compute log-likelihood
        #======================================================================
        
        logL = -(output_dim*(num_data*log_2_pi+logL_R+psi0-np.trace(LmInvPsi2LmInvT))+YRY- bbt)/2.-output_dim*logdet_L/2.

        #======================================================================
        # Compute dL_dKmm
        #======================================================================

        dL_dKmm =  dL_dpsi2R - output_dim* backsub_both_sides(Lm, LmInvPsi2LmInvT)/2 #LmInv.T.dot(LmInvPsi2LmInvT).dot(LmInv)/2.

        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================

        LLInvLmT = dtrtrs(LL, Lm.T)[0]
        cov = tdot(LLInvLmT.T)

        wd_inv = backsub_both_sides(Lm, np.eye(input_dim)- backsub_both_sides(LL, np.identity(input_dim), transpose='left'), transpose='left')
        post = Posterior(woodbury_inv=wd_inv, woodbury_vector=v.T, K=Kmm, mean=None, cov=cov, K_chol=Lm)

        #======================================================================
        # Compute dL_dthetaL for uncertian input and non-heter noise
        #======================================================================

        # if isinstance(likelihood, Gaussian_Gamma):
        #     from scipy.special import polygamma
        #     dL_dthetaL = ((YRY + output_dim*psi0)/2. - (dL_dpsi2R*psi2).sum() - np.trace(LLinvPsi1TYYTPsi1LLinvT))/-beta
        #     likelihood.q_a.gradient = num_data*output_dim/2.*polygamma(1, likelihood.q_a) + dL_dthetaL/likelihood.q_b
        #     likelihood.q_b.gradient = num_data*output_dim/(-2.*likelihood.q_b) +dL_dthetaL*(-likelihood.q_a/(likelihood.q_b*likelihood.q_b))
        # else:
        dL_dthetaL = (YRY*beta + beta*output_dim*psi0 - num_data*output_dim*beta)/2. - beta*(dL_dpsi2R*psi2).sum() - beta*np.trace(LLinvPsi1TYYTPsi1LLinvT)
        
        #======================================================================
        # Compute dL_dpsi
        #======================================================================

        dL_dpsi0 = -output_dim * (beta * np.ones((num_data,)))/2.

        if uncertain_outputs:
            m,s = Y.mean, Y.variance
            dL_dpsi1 = beta*(np.dot(m,v)+Shalf[:,None]*psi1SP)
        else:
            dL_dpsi1 = beta*np.dot(Y,v)

        if uncertain_inputs:
            dL_dpsi2 = beta* dL_dpsi2R
        else:
            dL_dpsi1 += np.dot(psi1,dL_dpsi2R)*2.
            dL_dpsi2 = None
        
        if uncertain_inputs:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dpsi0':dL_dpsi0,
                         'dL_dpsi1':dL_dpsi1,
                         'dL_dpsi2':dL_dpsi2,
                         'dL_dthetaL':dL_dthetaL}
        else:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dKdiag':dL_dpsi0,
                         'dL_dKnm':dL_dpsi1,
                         'dL_dthetaL':dL_dthetaL}
            
        if uncertain_outputs:
            m,s = Y.mean, Y.variance
            psi1LmiLLi = dtrtrs(LmLL, psi1.T)[0].T 
            LLiLmipsi1Y = b.T
            grad_dict['dL_dYmean'] = -m*beta+ psi1LmiLLi.dot(LLiLmipsi1Y)
            grad_dict['dL_dYvar'] = beta/-2.+ np.square(psi1LmiLLi).sum(axis=1)/2

        return post, logL, grad_dict



