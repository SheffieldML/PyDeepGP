# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import sys
import numpy as np
from scipy.linalg import LinAlgError
from GPy import Model,likelihoods
from GPy.core.parameterization.variational import VariationalPosterior, NormalPosterior
from ..layers import ObservedLayer, ObservedMRDLayer, HiddenLayer, TopHiddenLayer

class DeepGP(Model):
    """
    Deep Gaussian process model. 
    To use repeatX, you need to:
    a) Set dimensionality of intermediate kernels to [..]+Xtrain.xhape[1]
    b) Call the constructor with repeatX=True
    e.g.: 
          kern = [GPy.kern.RBF(Ds+Xtrain.shape[1], ARD=True), GPy.kern.RBF(Xtrain.shape[1], ARD=True)]
          m = deepgp.DeepGP(..., repeatX=True)
    """
    
    def __init__(self, nDims, Y, X=None, num_inducing=10,
                 likelihood = None, inits='PCA', name='deepgp',
                 kernels=None, obs_data='cont', back_constraint=True,
                 encoder_dims=None, mpi_comm=None, mpi_root=0, repeatX=False,
                 inference_method=None, **kwargs):
        super(DeepGP, self).__init__(name=name)
        self._IN_OPTIMIZATION_ = False
        
        # Back-forward compatibility with old interface
        if encoder_dims is None and 'MLP_dims' in kwargs:
            encoder_dims = kwargs['MLP_dims']
        
        self.mpi_comm, self.mpi_root = mpi_comm, mpi_root
        self.obs_data = obs_data
        self.nLayers = len(nDims)-1
        self.nDims = nDims
        self.input_dim = self.nDims[-1]

        # Manifold relevance determination
        mrd_flag = isinstance(Y,(list,tuple))
        if mrd_flag:
            assert isinstance(nDims[0], (list,tuple)), "The data dimension for outputs has to be a list or tuple for MRD!"
        else:
            self.output_dim = self.nDims[0]
            assert self.output_dim==Y.shape[1], "The dimensionality of output has to agree with the dimensionality of data (Y)!"

        # Back constraints
        self.back_constraint = back_constraint

        self.X_observed = X is not None

        if self.X_observed:
            assert self.input_dim==X.shape[1], "The dimensionality of output has to agree with the dimensionality of data (X)!"
            if mrd_flag:
                for i in range(len(Y)):
                    assert Y[i].shape[0]==X.shape[0], "The numbers of datapoints in X and Y have to be equal!"
            else:
                assert Y.shape[0]==X.shape[0], "The numbers of datapoints in X and Y have to be equal!"
        self.repeatX = repeatX
        self._log_marginal_likelihood = np.nan
        
        # self.Y = Y
        # self.X = X
        Xs = self._init_Xs(Y,X)
        if obs_data=='classification':
            self.nClasses = nDims[1]
            assert nDims[0]==1
            assert np.all(Y<self.nClasses) and np.all(Y>=0)
        elif obs_data=='binary':
            self.nClasses = 2
            assert nDims[0]==nDims[1]
            assert np.all(np.logical_or(Y==0, Y==1))
        elif obs_data!='cont':
            raise ValueError('obs_data has to be one of the following: "classification", "binary" or "cont"!')

        self.auto_update = auto_update = False if inference_method=='svi' else True

        # Parameters which exist differently per layer but specified as single componenents are here expanded to each layer
        if not isinstance(num_inducing, list or tuple): num_inducing = [num_inducing]*self.nLayers
        if not isinstance(inits, list or tuple): inits = [inits]*self.nLayers
        
        # Initialize Layers
        self.layers = []
        for i in range(self.nLayers):
            if i==0:
                if mrd_flag:#isinstance(nDims[0], (list,tuple)):
                    # MRD
                    #raise NotImplementedError()
                    self.layers.append(ObservedMRDLayer(nDims[0],
                                                        nDims[1], Y,
                                                        X=Xs[i], likelihood=likelihood,
                                                        num_inducing=num_inducing[i],
                                                        init=inits[i],
                                                        kernel=kernels[i] if kernels is not None else None,
                                                        back_constraint=back_constraint,
                                                        mpi_comm=mpi_comm,
                                                        mpi_root=mpi_root))
                else:
                    if self.X_observed and self.repeatX:
                        # self.layers.append(ObservedLayer(nDims[0],nDims[1], Y, X=Xs[i], likelihood=likelihood, num_inducing=num_inducing[i], init=inits[i], kernel=kernels[i] if kernels is not None else None, back_constraint=back_constraint, inference_method=inference_method, mpi_comm=mpi_comm, mpi_root=mpi_root))
                        self.layers.append(ObservedLayer(nDims[0],
                                                         Xs[i].shape[1],
                                                         Y, X=Xs[i], likelihood=likelihood,
                                                         num_inducing=num_inducing[i],
                                                         init=inits[i],
                                                         kernel=kernels[i] if kernels is not None else None,
                                                         back_constraint=back_constraint,
                                                         inference_method=inference_method,
                                                         mpi_comm=mpi_comm,
                                                         mpi_root=mpi_root,
                                                         auto_update=auto_update,
                                                         repeatX=True,
                                                         repeatXsplit=self.nDimsOrig[1]))
                        self.layers[-1].X_dim_free = range(self.nDimsOrig[1])
                        self.layers[-1].X_dim_top  = np.arange(self.nDimsOrig[1],
                                                               self.nDimsOrig[1]+X.shape[1]).tolist()
                    else:
                        self.layers.append(ObservedLayer(nDims[0],nDims[1],
                                                         Y, X=Xs[i], likelihood=likelihood,
                                                         num_inducing=num_inducing[i],
                                                         init=inits[i],
                                                         kernel=kernels[i] if kernels is not None else None,
                                                         back_constraint=back_constraint,
                                                         inference_method=inference_method,
                                                         mpi_comm=mpi_comm, mpi_root=mpi_root,
                                                         auto_update=auto_update))
            elif i==self.nLayers-1:
                if isinstance(X, VariationalPosterior):
                    X_variance, uncertain_inputs = X.variance.values, True
                else:
                    X_variance, uncertain_inputs = None, X is None
                self.layers.append(TopHiddenLayer(self.layers[-1],
                                                  nDims[i+1],
                                                  num_inducing=num_inducing[i],
                                                  X=Xs[i], X_variance=X_variance,
                                                  uncertain_inputs=uncertain_inputs,
                                                  name='layer_'+str(i),
                                                  init=inits[i],
                                                  kernel=kernels[i] if kernels is not None else None,
                                                  back_constraint=True if uncertain_inputs and back_constraint else False,
                                                  inference_method=inference_method,
                                                  mpi_comm=mpi_comm, mpi_root=mpi_root,
                                                  auto_update=auto_update))
            else:
                self.layers.append(HiddenLayer(self.layers[-1],
                                               nDims[i+1], X=Xs[i],
                                               num_inducing=num_inducing[i],
                                               name='layer_'+str(i), init=inits[i],
                                               kernel=kernels[i] if kernels is not None else None,
                                               back_constraint=back_constraint,
                                               inference_method=inference_method,
                                               mpi_comm=mpi_comm, mpi_root=mpi_root,
                                               auto_update=auto_update)) 
        if self.nLayers==1:
            self.layers[0].set_as_toplayer()
        
        if self.back_constraint:
            from ..layers import EncoderLayer 
            direction = 'top_down' if self.X_observed else 'bottom_up'
            if self.X_observed:
                self.enclayers = [EncoderLayer(self.layers[i],
                                               direction=direction,
                                               encoder_dims=encoder_dims[i-1] if encoder_dims is not None else None,
                                               mpi_comm = mpi_comm,
                                               mpi_root=mpi_root,
                                               name='enclayer_'+str(i)) for i in range(self.nLayers-1,0,-1)]
            else:
                self.enclayers = [ EncoderLayer(self.layers[i], direction=direction, encoder_dims=encoder_dims[i] if encoder_dims is not None else None, mpi_comm = mpi_comm, mpi_root=mpi_root,
                                            name='enclayer_'+str(i)) for i in range(self.nLayers)]
            self.link_parameters(*(self.enclayers+self.layers))
        else:
            self.link_parameters(*self.layers)
        
        if self.X_observed and self.repeatX:
            self.ensure_repeatX_constraints()
        if self.mpi_comm is not None:
            from deepgp.util.parallel import broadcastArrays
            broadcastArrays([self.param_array], self.mpi_comm, self.mpi_root)

    @property
    def Y(self):
        return self.layers[0].Y

    @property
    def X(self):
        return self.layers[-1].X

    def _init_Xs(self, Ya, Xa):
        
        if isinstance(Ya,list):
            return [None for d in self.nDims[1:]]
        else:
            nDims, N = self.nDims, Ya.shape[0]
        if self.back_constraint:
            if self.X_observed:
                Xs = [np.random.rand(N,d) for d in nDims[1:-1]]+[Xa]
            else:
                Xs = [np.random.rand(N,d) for d in nDims[1:]]
        elif self.X_observed and len(nDims)==3:
            X = Xa.mean.values if isinstance(Xa, VariationalPosterior) else Xa
            from deepgp.util.util import initialize_latent
            X_mid = initialize_latent('PCA', N, nDims[1], X)[0]
            if X.shape[1]<self.input_dim:
                tmp = np.random.randn(*X.shape)
                tmp[:,:X_mid.shape[1]] = X_mid
                X_mid = tmp
            if self.repeatX:
                if isinstance(X, VariationalPosterior):
                    #-- Haven't tested this case
                    # ...
                    raise NotImplementedError()
                else:
                    X_mid2 = np.hstack((X_mid, X))
                    # Should we re-normalize everything????
                    Xmean, Xstd = X_mid2.mean(0), X_mid2.std(0)+1e-20
                    X_mid2 -= Xmean[np.newaxis,:]
                    X_mid2 /= Xstd[np.newaxis,:]

                    self.repeatX_Xmean = Xmean.copy()[X_mid.shape[1]:]
                    self.repeatX_Xstd = Xstd.copy()[X_mid.shape[1]:]
    
                    self.nDimsOrig = nDims[:]
                    nDims[1] = X_mid2.shape[1]
                    X_mid = X_mid2
            Xs = [X_mid, X]
        elif self.X_observed:
            X = Xa.mean.values if isinstance(Xa, VariationalPosterior) else Xa
            Xs = [None for d in nDims[1:-1]]+[X]
        else:
            Xs = [None for d in nDims[1:]]
        return Xs
    
    def ensure_repeatX_constraints(self):
        for i in range(self.nLayers):
            if hasattr(self.layers[i],'X_dim_top'):
                self.layers[i].X.mean[:,self.layers[i].X_dim_top].fix(warning=False)
                self.layers[i].X.variance[:,self.layers[i].X_dim_top] = 1e-6*np.ones((self.layers[i].X.variance[:,self.layers[i].X_dim_top].shape[0],self.layers[i].X.variance[:,self.layers[i].X_dim_top].shape[1]))
                self.layers[i].X.variance[:,self.layers[i].X_dim_top].fix(warning=False)

    def log_likelihood(self):
        return self._log_marginal_likelihood
        
    def parameters_changed(self):
        if not self.auto_update: [l.update_layer() for l in self.layers]
        self._log_marginal_likelihood = np.sum([l._log_marginal_likelihood for l in self.layers])
        if self.back_constraint:
            [l.update_gradients() for l in self.enclayers[::-1]]
            if self.mpi_comm is not None:
                [l._gather_gradients() for l in self.enclayers]
        
    def add_layer(self, dim_up, X=None,
                  X_variance=None, Z=None,
                  uncertain_inputs=True, num_inducing=10,
                  kernel=None, inference_method=None,
                  noise_var=1e-2, init='rand', MLP_dims=None,
                  back_constraint=True,  name='tophiddenlayer'):
        """Add a layer"""
        ls = []
        if isinstance(self.layers[-1], TopHiddenLayer):
            layer = HiddenLayer.from_TopHiddenLayer(self.layers[-1],
                                                    name='layer_'+str(self.nLayers-1))
            self.unlink_parameter(self.layers[-1])
            self.layers.pop(-1)
            self.layers.append(layer)
            ls.append(layer)
        elif isinstance(self.layers[-1], ObservedLayer):
            self.layers[0].set_as_toplayer(False)
            
        self.layers.append(TopHiddenLayer(self.layers[-1], dim_up,
                                          num_inducing=num_inducing,
                                          name='layer_'+str(self.nLayers), init=init, 
                                          kernel=kernel, noise_var=noise_var,
                                          MLP_dims=MLP_dims, mpi_comm=self.mpi_comm,
                                          mpi_root=self.mpi_root,
                                          back_constraint=back_constraint, X=X,
                                          X_variance=X_variance, Z=Z,
                                          uncertain_inputs=uncertain_inputs))
        ls.append(self.layers[-1])
        self.nLayers += 1
        self.nDims.append(dim_up)
        self.input_dim = self.nDims[-1]
        self.link_parameters(*ls)
        
    def gen_pred_model(self, Y=None, init_X='encoder', binObserved=False):
        from GPy.core.parameterization.variational import NormalPosterior
        from deepgp.models.pred_model import PredDeepGP
        
        if Y is not None:
            Xs = [Y]
            
            if init_X=='nn':
                xy = self.collect_all_XY()
                for i in range(self.nLayers):
                    x_mean, x_var = PredDeepGP._init_Xs_NN(Y,i,xy)
                    Xs.append(NormalPosterior(x_mean, x_var))
            elif init_X=='encoder':
                x_mean = Y
                x_mean[np.isnan(x_mean)] = 0.
                for layer in self.layers:
                    x_mean = layer.encoder.predict(x_mean)
                    Xs.append(NormalPosterior(x_mean, np.ones(x_mean.shape)*layer.X_var))
        
        layers = []
        layer_lower = None
        for i in range(self.nLayers):
            layers.append(self.layers[i].gen_pred_layer(layer_lower=layer_lower,Y=Xs[i], X=Xs[i+1], binObserved=(i==0 and binObserved)))
            layer_lower = layers[-1]
        
        pmodel = PredDeepGP(layers)
        if init_X=='nn': pmodel.set_train_data(xy)
        return pmodel
    
    def collect_all_XY(self, root=0):
        if self.mpi_comm is None:
            XY = [self.obslayer.Y.copy()]
            for l in self.layers: XY.append(l.X.copy())
            return XY
        else:
            from mpi4py import MPI
            from GPy.core.parameterization.variational import NormalPosterior
            N,D = self.Y.shape
            N_list = np.array(self.mpi_comm.allgather(N))
            N_all = np.sum(N_list)
            Y_all = np.empty((N_all,D)) if self.mpi_comm.rank==root else None
            self.mpi_comm.Gatherv([self.Y, MPI.DOUBLE], [Y_all, (N_list*D, None), MPI.DOUBLE], root=root)
            if self.mpi_comm.rank==root:
                XY = [Y_all]
            for l in self.layers:
                Q = l.X.shape[1]
                X_mean_all =  np.empty((N_all,Q)) if self.mpi_comm.rank==root else None
                self.mpi_comm.Gatherv([l.X.mean.values, MPI.DOUBLE], [X_mean_all, (N_list*Q, None), MPI.DOUBLE], root=root)
                X_var_all =  np.empty((N_all,Q)) if self.mpi_comm.rank==root else None
                self.mpi_comm.Gatherv([l.X.variance.values, MPI.DOUBLE], [X_var_all, (N_list*Q, None), MPI.DOUBLE], root=root)
                if self.mpi_comm.rank==root:
                    XY.append(NormalPosterior(X_mean_all, X_var_all))
            if self.mpi_comm.rank==root: return XY
            else: return None
        
    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None):
        """Make a prediction from the deep Gaussian process model for a given input"""
        from GPy.core.parameterization.variational import NormalPosterior
        
        if self.repeatX:
            assert self.nLayers==2
            mean,var = self.layers[-1].predict(Xnew)
            Xnew_norm = (Xnew - self.repeatX_Xmean)/self.repeatX_Xstd
            Xmean = np.hstack([mean,Xnew_norm])
            Xvar = np.empty_like(Xmean)
            Xvar[:] = 1e-6
            Xvar[:,:self.nDimsOrig[1]] = var
            x = NormalPosterior(Xmean,Xvar)
        else:
            x = Xnew
            for l in self.layers[:0:-1]:
                mean, var = l.predict(x)
                var = np.clip(var,1e-8, np.inf)
                if var.shape[1]==1:
                    var = np.tile(var,mean.shape[1])
                x = NormalPosterior(mean, var)
        return self.layers[0].predict(x)

    @Model.optimizer_array.setter
    def optimizer_array(self, p):
        if self.mpi_comm != None:
            if self._IN_OPTIMIZATION_ and self.mpi_comm.rank==self.mpi_root:
                self.mpi_comm.Bcast(np.int32(1),root=self.mpi_root)
            self.mpi_comm.Bcast(p, root=self.mpi_root)
        Model.optimizer_array.fset(self,p)

    def optimize(self, optimizer=None, start=None, **kwargs):
        self._IN_OPTIMIZATION_ = True
        if self.mpi_comm is None:
            super(DeepGP, self).optimize(optimizer,start,**kwargs)
        elif self.mpi_comm.rank==self.mpi_root:
            super(DeepGP, self).optimize(optimizer,start,**kwargs)
            self.mpi_comm.Bcast(np.int32(-1),root=self.mpi_root)
        elif self.mpi_comm.rank!=self.mpi_root:
            x = self.optimizer_array.copy()
            flag = np.empty(1,dtype=np.int32)
            while True:
                self.mpi_comm.Bcast(flag,root=self.mpi_root)
                if flag==1:
                    try:
                        self.optimizer_array = x
                    except (LinAlgError, ZeroDivisionError, ValueError):
                        pass
                elif flag==-1:
                    break
                else:
                    self._IN_OPTIMIZATION_ = False
                    raise Exception("Unrecognizable flag for synchronization!")
        self._IN_OPTIMIZATION_ = False
        
    def predict_withSamples(self, X, nSamples=100):
        
        y_mean, y_var = self.layers[-1].predict(X)
        y_samples = np.random.randn(nSamples,X.shape[0],y_mean.shape[1])*np.sqrt(y_var)+y_mean
        for l in self.layers[-2::-1]:
            y_mean, y_var = l.predict(y_samples.reshape(nSamples*X.shape[0],-1))
            y_samples = np.random.randn(*y_var.shape)*np.sqrt(y_var)+y_mean
        y_samples = y_samples.reshape(nSamples, X.shape[0],-1)
        y_mean = y_samples.mean(0)
        y_var = y_samples.var(0)
        return y_mean, y_var
        
    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None):
        """
        Get the predictive quantiles around the prediction at X

        :param X: The points at which to make a prediction
        :type X: np.ndarray (Xnew x self.input_dim)
        :param quantiles: tuple of quantiles, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :returns: list of quantiles for each X and predictive quantiles for interval combination
        :rtype: [np.ndarray (Xnew x self.output_dim), np.ndarray (Xnew x self.output_dim)]
        """
        m, v = self.predict(X,  full_cov=False)
        return self.layers[0].likelihood.predictive_quantiles(m, v-self.layers[0].likelihood.variance, quantiles, Y_metadata=Y_metadata)

    def append_XY(self, Y, X):
        from GPy import ObsAr
        assert self.X_observed
        self.update_model(False)
        self.layers[-1].set_newX(X,append=True)
        for l in self.layers[:0:-1]:
            y_mean, y_var = l.predict(l.X[-X.shape[0]:])
            l.layer_lower.set_newX(NormalPosterior(y_mean, np.ones_like(y_mean)*y_var), append=True)
        self.layers[0].Y = ObsAr(np.vstack([self.layers[0].Y, Y]))
        self.update_model(True)
