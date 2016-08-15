
from GPy.core import Parameterized
from GPy import Param
from GPy.core.parameterization.transformations import Logexp
from GPy.core.parameterization.variational import VariationalPosterior


class EncoderLayer(Parameterized):
    """
    Layer class for the recognition model
    
    :param layer: the corresponding layer in DeepGP
    :param direction: the direction of the applied encoder: 'bottom_up' or 'top_down'
    :param encoder: the choice of encoder. the current supported encoders: 'mlp'
    :param encoder_dims: the number of units in the hidden layers
    """
    
    def __init__(self, layer, direction='bottom_up', encoder='mlp', encoder_dims=None, mpi_comm=None, mpi_root=0, name='encoder'):
        super(EncoderLayer, self).__init__(name=name)
        self.mpi_comm, self.mpi_root = mpi_comm, mpi_root
        self.layer = layer
        if direction=='bottom_up':
            self.bottom_up = True
            self.X, self.Y = layer.Y, layer.X
        elif direction=='top_down':
            self.bottom_up = False
            self.X, self.Y = layer.X, layer.Y
        else:
            raise Exception('the argument of "direction" has to be either "bottom_up" or "top_down"!')
        self.uncertain_input = isinstance(self.X, VariationalPosterior)
        assert isinstance(self.Y, VariationalPosterior), "No need to have a encoder layer for certain output!"
        
        if encoder=='mlp':
            dim_in, dim_out = self.X.shape[1], self.Y.shape[1]
            from copy import deepcopy
            from deepgp.encoder.mlp import MLP
            self.encoder = MLP([dim_in, int((dim_in+dim_out)*2./3.), int((dim_in+dim_out)/3.), dim_out] if encoder_dims is None \
                               else [dim_in]+deepcopy(encoder_dims)+[dim_out])
        else:
            raise Exception('Unsupported encoder type: '+encoder)
        self.Y_var_common = Param('Y_var', self.Y.variance.values[1].copy(),Logexp())
        
        # Synchronize across MPI nodes
        if self.mpi_comm is not None:
            from ..util.parallel import broadcastArrays
            broadcastArrays([self.encoder.param_array, self.Y_var_common], self.mpi_comm, self.mpi_root)
        self.link_parameters(self.encoder, self.Y_var_common)
    
    @property
    def _X_vals(self):
        return self.X.mean.values if self.uncertain_input else self.X
        
    def parameters_changed(self):
        self.Y.mean[:] = self.encoder.predict(self._X_vals)
        self.Y.variance[:] = self.Y_var_common.values
        
    def update_gradients(self):
        dL_dY = self.Y.mean.gradient
        self.Y_var_common.gradient[:] = self.Y.variance.gradient.sum(axis=0)
        X_grad = self.encoder.update_gradient(self._X_vals, dL_dY)
        if self.uncertain_input:
            self.X.mean.gradient += X_grad
            
    def _gather_gradients(self):
        from deepgp.util.parallel import numpy_to_MPI_typemap
        t = numpy_to_MPI_typemap(self.encoder.gradient.dtype)
        self.mpi_comm.Reduce([self.encoder.gradient.copy(), t], [self.encoder.gradient, t], root=self.mpi_root)
        
        t = numpy_to_MPI_typemap(self.Y_var.gradient.dtype)
        self.mpi_comm.Reduce([self.Y_var.gradient.copy(), t], [self.Y_var.gradient, t], root=self.mpi_root)
