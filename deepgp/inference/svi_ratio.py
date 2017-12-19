


#from .posterior import Posterior
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

class SVI_Ratio(LatentFunctionInference):
    """
    Inference the marginal likelihood through \frac{p(y,y*)}{p(y)}
    """
    const_jitter = 1e-6
    def __init__(self, mpi_comm=None):

        self.mpi_comm = mpi_comm
    
    def get_trYYT(self, Y):
        return np.sum(np.square(Y))

    def get_YYTfactor(self, Y):
        N, D = Y.shape
        if (N>=D):
            return Y.view(np.ndarray)
        else:
            return jitchol(tdot(Y))

    def gatherPsiStat(self, kern, X, Z, Y, beta, uncertain_inputs, D, missing_data):

        assert beta.size == 1

        if uncertain_inputs:
            psi0 = kern.psi0(Z, X)
            psi1 = kern.psi1(Z, X)*beta
            psi2 = kern.psi2(Z, X)*beta if not missing_data else kern.psi2n(Z, X)*beta
        else:
            psi0 = kern.Kdiag(X)
            psi1 = kern.K(X, Z)
            if missing_data:
                psi2 = psi1[:,None,:]*psi1[:,:,None]*beta
            else:
                psi2 = tdot(psi1.T)*beta
            psi1 = psi1*beta
            
        if isinstance(Y, VariationalPosterior):
            m, s = Y.mean, Y.variance
            psi1Y = np.dot(m.T,psi1) # DxM            
            YRY = (np.square(m).sum()+s.sum())*beta
            psi0 = (D*psi0).sum()*beta
        elif missing_data:
            psi1Y = np.dot((Y).T,psi1) # DxM
            trYYT = self.get_trYYT(Y)
            YRY = trYYT*beta
            psi0 = (psi0*D).sum()*beta
        else:
            psi1Y = np.dot(Y.T,psi1) # DxM
            trYYT = self.get_trYYT(Y)
            YRY = trYYT*beta
            psi0 = (psi0*D).sum()*beta
            
        return psi0, psi2, YRY, psi1, psi1Y

    def inference(self, kern, X, Z, likelihood, Y, qU):
        """
        The SVI-VarDTC inference
        """

        if isinstance(Y, np.ndarray) and np.any(np.isnan(Y)):
            missing_data = True
            N, M, Q = Y.shape[0], Z.shape[0], Z.shape[1]
            Ds = Y.shape[1] - (np.isnan(Y)*1).sum(1)
            Ymask = 1-np.isnan(Y)*1
            Y_masked = np.zeros_like(Y)
            Y_masked[Ymask==1] = Y[Ymask==1]
            ND = Ymask.sum()
        else:
            missing_data = False
            N, D, M, Q = Y.shape[0], Y.shape[1], Z.shape[0], Z.shape[1]
            ND = N*D

        uncertain_inputs = isinstance(X, VariationalPosterior)
        uncertain_outputs = isinstance(Y, VariationalPosterior)

        beta = 1./np.fmax(likelihood.variance, 1e-6)

        psi0, psi2, YRY, psi1, psi1Y = self.gatherPsiStat(kern, X, Z, Y if not missing_data else Y_masked, beta, uncertain_inputs, D if not missing_data else Ds, missing_data)
        
        #======================================================================
        # Compute Common Components
        #======================================================================
        
        mu, S = qU.mean, qU.covariance
        mupsi1Y = mu.dot(psi1Y)

        Kmm = kern.K(Z).copy()
        diag.add(Kmm, self.const_jitter)
        Lm = jitchol(Kmm)
        
        if missing_data:
            S_mu = S[None,:,:]+mu.T[:,:,None]*mu.T[:,None,:]
            NS_mu = S_mu.T.dot(Ymask.T).T
            LmInv = dtrtri(Lm)
            
            LmInvPsi2LmInvT = np.swapaxes(psi2.dot(LmInv.T),1,2).dot(LmInv.T)            
            LmInvSmuLmInvT =  np.swapaxes(NS_mu.dot(LmInv.T),1,2).dot(LmInv.T)
            
            B = mupsi1Y+ mupsi1Y.T +(Ds[:,None,None]*psi2).sum(0)
            tmp = backsub_both_sides(Lm, B,'right')
            
            logL =  -ND*log_2_pi/2. +ND*np.log(beta)/2. - psi0/2. - YRY/2.  \
                       -(LmInvSmuLmInvT*LmInvPsi2LmInvT).sum()/2. +np.trace(tmp)/2.
        else:
            S_mu = S*D+tdot(mu)
            if uncertain_inputs:
                LmInvPsi2LmInvT = backsub_both_sides(Lm, psi2, 'right')
            else:
                LmInvPsi2LmInvT = tdot(dtrtrs(Lm, psi1.T)[0])/beta #tdot(psi1.dot(LmInv.T).T) /beta        
            LmInvSmuLmInvT = backsub_both_sides(Lm, S_mu, 'right')
            
            B = mupsi1Y+ mupsi1Y.T +D*psi2
            tmp = backsub_both_sides(Lm, B,'right')
            
            logL =  -ND*log_2_pi/2. +ND*np.log(beta)/2. - psi0/2. - YRY/2.  \
                       -(LmInvSmuLmInvT*LmInvPsi2LmInvT).sum()/2. +np.trace(tmp)/2.

        #======================================================================
        # Compute dL_dKmm
        #======================================================================

        dL_dKmm = np.eye(M)

        #======================================================================
        # Compute dL_dthetaL for uncertian input and non-heter noise
        #======================================================================

        dL_dthetaL = None #(YRY*beta + beta*output_dim*psi0 - num_data*output_dim*beta)/2. - beta*(dL_dpsi2R*psi2).sum() - beta*np.trace(LLinvPsi1TYYTPsi1LLinvT)
        
        #======================================================================
        # Compute dL_dpsi
        #======================================================================

        if missing_data:
            dL_dpsi0 = -Ds * (beta * np.ones((N,)))/2.
        else:
            dL_dpsi0 = -D * (beta * np.ones((N,)))/2.

        if uncertain_outputs:
            Ym,Ys = Y.mean, Y.variance
            dL_dpsi1 = dtrtrs(Lm, dtrtrs(Lm, Ym.dot(mu.T).T)[0], trans=1)[0].T*beta
        else:
            if missing_data:
                dL_dpsi1 = dtrtrs(Lm, dtrtrs(Lm, (Y_masked).dot(mu.T).T)[0], trans=1)[0].T*beta
            else:
                dL_dpsi1 = dtrtrs(Lm, dtrtrs(Lm, Y.dot(mu.T).T)[0], trans=1)[0].T*beta

        if uncertain_inputs:
            if missing_data:
                dL_dpsi2 = np.swapaxes((Ds[:,None,None]*np.eye(M)[None,:,:]-LmInvSmuLmInvT).dot(LmInv),1,2).dot(LmInv)*beta/2.
            else:
                dL_dpsi2 = beta*backsub_both_sides(Lm, D*np.eye(M)-LmInvSmuLmInvT, 'left')/2.
        else:
            dL_dpsi1 += beta*psi1.dot(dL_dpsi2+dL_dpsi2.T) 
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
            Ym = Y.mean
            grad_dict['dL_dYmean'] = -Ym*beta+ dtrtrs(Lm,psi1.T)[0].T.dot(dtrtrs(Lm,mu)[0])
            grad_dict['dL_dYvar'] = beta/-2.

        return logL, grad_dict



