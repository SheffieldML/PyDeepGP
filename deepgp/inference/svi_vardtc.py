


#from .posterior import Posterior
from GPy.util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri,pdinv, dpotri
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

class SVI_VarDTC(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    const_jitter = 1e-6
    def __init__(self, mpi_comm=None, mpi_root=0):
        self.mpi_comm = mpi_comm
        self.mpi_root=mpi_root
    
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
            psi0 = kern.psi0(Z, X).sum()*beta
            psi1 = kern.psi1(Z, X)*beta
            psi2 = kern.psi2(Z, X)*beta
        else:
            psi0 = kern.Kdiag(X).sum()*beta
            psi1 = kern.K(X, Z)*beta
            psi2 = None
            
        if isinstance(Y, VariationalPosterior):
            m, s = Y.mean, Y.variance
            psi1Y = np.dot(m.T,psi1) # DxM
            YRY = (np.square(m).sum()+s.sum())*beta
        else:
            psi1Y = np.dot(Y.T,psi1) # DxM            
            trYYT = self.get_trYYT(Y)
            YRY = trYYT*beta
            
        if self.mpi_comm is not None:
            from ..util.parallel import allReduceArrays
            psi0, psi2, YRY, psi1Y = allReduceArrays([psi0, psi2, YRY, psi1Y], self.mpi_comm)
        return psi0, psi2, YRY, psi1, psi1Y

    def inference(self, kern, X, Z, likelihood, Y, qU_mean ,qU_var, Kuu_sigma=None):
        """
        The SVI-VarDTC inference
        """

        N, D, M, Q = Y.shape[0], Y.shape[1], Z.shape[0], Z.shape[1]

        uncertain_inputs = isinstance(X, VariationalPosterior)
        uncertain_outputs = isinstance(Y, VariationalPosterior)

        beta = 1./likelihood.variance

        psi0, psi2, YRY, psi1, psi1Y = self.gatherPsiStat(kern, X, Z, Y, beta, uncertain_inputs)
        
        #======================================================================
        # Compute Common Components
        #======================================================================

        Kuu = kern.K(Z).copy()
        if Kuu_sigma is not None:
            diag.add(Kuu, Kuu_sigma)
        else:
            diag.add(Kuu, self.const_jitter)
        Lm = jitchol(Kuu)
        
        mu, S = qU_mean, qU_var
        Ls = jitchol(S)
        LinvLs = dtrtrs(Lm, Ls)[0]
        Linvmu = dtrtrs(Lm, mu)[0]
        psi1YLinvT = dtrtrs(Lm,psi1Y.T)[0].T
        
        self.mid = {
                    'qU_L': Ls,
                    'LinvLu': LinvLs,
                    'L':Lm,
                    'Linvmu': Linvmu}
        
        if uncertain_inputs:
            LmInvPsi2LmInvT = backsub_both_sides(Lm, psi2, 'right')
        else:
            LmInvPsi2LmInvT = tdot(dtrtrs(Lm, psi1.T)[0])/beta 
        
        LmInvSmuLmInvT = tdot(LinvLs)*D+tdot(Linvmu)
        
#         logdet_L = np.sum(np.log(np.diag(Lm)))
#         logdet_S = np.sum(np.log(np.diag(Ls)))
        
        #======================================================================
        # Compute log-likelihood
        #======================================================================
        
        logL_R = -N*np.log(beta)
        logL = -N*D*log_2_pi/2. -D*logL_R/2. - D*psi0/2. - YRY/2.  \
                     -(LmInvSmuLmInvT*LmInvPsi2LmInvT).sum()/2. + np.trace(LmInvPsi2LmInvT)*D/2.+(Linvmu*psi1YLinvT.T).sum()
                
        #======================================================================
        # Compute dL_dKmm
        #======================================================================

        tmp1 = backsub_both_sides(Lm,LmInvSmuLmInvT.dot(LmInvPsi2LmInvT), 'left')
        tmp2 = Linvmu.dot(psi1YLinvT)
        tmp3 = backsub_both_sides(Lm,  - D*LmInvPsi2LmInvT  -tmp2-tmp2.T, 'left')/2.

        dL_dKmm = (tmp1+tmp1.T)/2. + tmp3

        #======================================================================
        # Compute dL_dthetaL for uncertian input and non-heter noise
        #======================================================================

        dL_dthetaL = -D*N*beta/2. -(- D*psi0/2. - YRY/2.-(LmInvSmuLmInvT*LmInvPsi2LmInvT).sum()/2. + np.trace(LmInvPsi2LmInvT)*D/2.+(Linvmu*psi1YLinvT.T).sum())*beta
        
        #======================================================================
        # Compute dL_dqU
        #======================================================================
        
        tmp1 = backsub_both_sides(Lm, - LmInvPsi2LmInvT, 'left')
        dL_dqU_mean = tmp1.dot(mu) + dtrtrs(Lm, psi1YLinvT.T,trans=1)[0]
        dL_dqU_var = D/2.*tmp1
        
        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================

        KuuInvmu = dtrtrs(Lm, Linvmu, trans=1)[0]
        tmp = backsub_both_sides(Lm,  np.eye(M) - tdot(LinvLs), 'left')

        post = Posterior(woodbury_inv=tmp, woodbury_vector=KuuInvmu, K=Kuu, mean=mu, cov=S, K_chol=Lm)
        
        #======================================================================
        # Compute dL_dpsi
        #======================================================================

        dL_dpsi0 = -D * (beta * np.ones((N,)))/2.

        if uncertain_outputs:
            dL_dpsi1 = Y.mean.dot(dtrtrs(Lm,Linvmu,trans=1)[0].T)*beta
        else:
            dL_dpsi1 = Y.dot(dtrtrs(Lm,Linvmu,trans=1)[0].T)*beta

        dL_dpsi2 = beta*backsub_both_sides(Lm, D*np.eye(M)-LmInvSmuLmInvT, 'left')/2.
        if not uncertain_inputs:
            dL_dpsi1 += psi1.dot(dL_dpsi2+dL_dpsi2.T)/beta
            dL_dpsi2 = None
            
        if uncertain_inputs:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dpsi0':dL_dpsi0,
                         'dL_dpsi1':dL_dpsi1,
                         'dL_dpsi2':dL_dpsi2,
                         'dL_dthetaL':dL_dthetaL,
                         'dL_dqU_mean':dL_dqU_mean,
                         'dL_dqU_var':dL_dqU_var}
        else:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dKdiag':dL_dpsi0,
                         'dL_dKnm':dL_dpsi1,
                         'dL_dthetaL':dL_dthetaL,
                         'dL_dqU_mean':dL_dqU_mean,
                         'dL_dqU_var':dL_dqU_var}

        if uncertain_outputs:
            m,s = Y.mean, Y.variance
            grad_dict['dL_dYmean'] = -m*beta+ dtrtrs(Lm,psi1.T)[0].T.dot(dtrtrs(Lm,mu)[0])
            grad_dict['dL_dYvar'] = beta/-2.

        return post, logL, grad_dict

    def comp_KL_qU(self, qU_mean ,qU_var):
        M,D = qU_mean.shape[0], qU_mean.shape[1]

        qU_L = self.mid['qU_L']
        L = self.mid['L']
        Linvmu = self.mid['Linvmu']
        LinvLu = self.mid['LinvLu']
        KuuInv = dpotri(L, lower=1)[0]
        
        Lu = qU_L
        LuInv = dtrtri(Lu)
        
        KL = D*M/-2. - np.log(np.diag(Lu)).sum()*D +np.log(np.diag(L)).sum()*D + np.square(LinvLu).sum()/2.*D + np.square(Linvmu).sum()/2.
        
        dKL_dqU_mean = dtrtrs(L, Linvmu, trans=True)[0] 
        dKL_dqU_var = (tdot(LuInv.T)/-2. +  KuuInv/2.)*D
        dKL_dKuu = KuuInv*D/2. -KuuInv.dot( tdot(qU_mean)+qU_var*D).dot(KuuInv)/2.

        return float(KL), dKL_dqU_mean, dKL_dqU_var, dKL_dKuu


