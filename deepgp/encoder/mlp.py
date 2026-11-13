# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
Multilayer perceptron encoder in plain NumPy (drop-in replacement for the
former Theano implementation; same classes, parameters and gradients).

Behaviour notes preserved from the Theano version:
- gradients are accumulated between ``prepare()`` calls;
- when the gradient is supplied externally (``update_gradient``), the L1/L2
  ``regularization`` term is not added (the Theano graph dropped ``add_cost``
  on that branch).
"""

from GPy.core import Model, Parameterized, Param
import numpy as np


def _softplus(x):
    return np.logaddexp(0.0, x)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Layer(Parameterized):

    def __init__(self, dim_up, dim_down, activation, regularization=None, reg_weight=0, W=None, b=None, name='layer'):
        super(Layer, self).__init__(name=name)
        self.dim_down = dim_down
        self.dim_up = dim_up
        self.layer_forward = None
        self.layer_backward = None
        self.activation = None if activation is None else activation.lower()
        if self.activation not in (None, 'tanh', 'softplus'):
            raise ValueError('Unsupported activation function: %r' % activation)
        self.regularization = regularization
        self.reg_weight = reg_weight

        if W is None:
            W = np.random.rand(dim_down, dim_up) * 2 - 1
            W *= np.sqrt(6. / (dim_up + dim_down))
        if b is None:
            b = np.zeros((dim_down,))
        self.W = Param('W', W)
        self.b = Param('b', b)
        self.link_parameters(self.W, self.b)
        self._W_acc = np.zeros_like(self.W.values)
        self._b_acc = np.zeros_like(self.b.values)

    def link_layers(self, layer_backward, layer_forward):
        self.layer_forward = layer_forward
        self.layer_backward = layer_backward

    def _prepare(self):
        self._W_acc[:] = 0.
        self._b_acc[:] = 0.
        if self.layer_forward is not None:
            self.layer_forward._prepare()

    def _update_gradient(self):
        self.W.gradient[:] = self._W_acc
        self.b.gradient[:] = self._b_acc
        if self.layer_forward is not None:
            self.layer_forward._update_gradient()

    def _forward(self, X):
        lin = X.dot(self.W.values.T) + self.b.values[None, :]
        if self.activation == 'tanh':
            out = np.tanh(lin)
        elif self.activation == 'softplus':
            out = _softplus(lin)
        else:
            out = lin
        return lin, out

    def _dact(self, lin, out):
        if self.activation == 'tanh':
            return 1. - out ** 2
        if self.activation == 'softplus':
            return _sigmoid(lin)
        return None

    def forward_all(self, X):
        cache, layer, h = [], self, X
        while layer is not None:
            lin, out = layer._forward(h)
            cache.append((layer, h, lin, out))
            h, layer = out, layer.layer_forward
        return h, cache

    def backward_all(self, cache, dout):
        for layer, h_in, lin, out in reversed(cache):
            d = layer._dact(lin, out)
            dlin = dout if d is None else dout * d
            layer._W_acc += dlin.T.dot(h_in)
            layer._b_acc += dlin.sum(0)
            dout = dlin.dot(layer.W.values)
        return dout


class MLP(Parameterized):
    """Multi-Layer Perceptron Class"""

    @staticmethod
    def clone(mlp):
        mlp_new = MLP(mlp.nUnits, activation=mlp.layers[0].activation, X_center=mlp.X_center,
                      regularization=mlp.regularization, reg_weight=mlp.reg_weight, positive_obs=mlp.positive_obs, name=mlp.name)
        mlp_new.param_array[:] = mlp.param_array
        mlp_new._trigger_params_changed()
        return mlp_new

    def __init__(self, nUnits, activation='tanh', X_center=None, regularization=None, reg_weight=0, positive_obs=False, name='mlp'):
        super(MLP, self).__init__(name=name)
        self.Y = None
        self.X = None
        self.nLayers = len(nUnits) - 1
        self.nUnits = nUnits
        self.X_center = X_center
        self.regularization = regularization
        self.reg_weight = reg_weight
        self.positive_obs = positive_obs
        self._warned_unused_regularization = False
        act = None if activation is None else activation.lower()
        if act not in (None, 'tanh', 'softplus'):
            raise ValueError('Unsupported activation function: %r' % activation)
        if regularization is not None:
            import warnings
            warnings.warn(
                "MLP regularization=%r is accepted for API compatibility but has "
                "no effect when gradients are supplied via update_gradient "
                "(historical Theano behaviour; see CIP-0001)." % (regularization,),
                FutureWarning,
                stacklevel=2,
            )
            self._warned_unused_regularization = True
        self.layers = [Layer(nUnits[i], nUnits[i + 1], activation=act if i != self.nLayers - 1 else ('softplus' if positive_obs else None),
                             regularization=regularization, reg_weight=reg_weight, name='layer_' + str(i + 1)) for i in range(self.nLayers)]
        for i in range(self.nLayers):
            self.layers[i].link_layers(self.layers[i - 1] if i > 0 else None, self.layers[i + 1] if i < self.nLayers - 1 else None)
        self.link_parameters(*self.layers)

    def update_gradient(self, X, dL):
        # External-gradient path: L1/L2 regularization remains a no-op (CIP-0001 R2).
        X = X if self.X_center is None else X - self.X_center
        if len(X.shape) == 1:
            X = X[None, :]
            dL = dL[None, :]
        _, cache = self.layers[0].forward_all(X)
        X_grad = self.layers[0].backward_all(cache, dL)
        self.layers[0]._update_gradient()
        return X_grad

    def prepare(self):
        """Clear accumulated gradients (call before updating gradients)."""
        self.layers[0]._prepare()

    def predict(self, X):
        X = X if self.X_center is None else X - self.X_center
        self.prepare()
        if len(X.shape) == 1:
            X = X[None, :]
        return self.layers[0].forward_all(X)[0]


class MLP_model(Model):
    """
    The Multi-layer Perceptron (MLP) model with L2 cost function.
    """
    def __init__(self, inputs, outputs, nUnits, activation='tanh', X_center=None, regularization=None, reg_weight=0, positive_obs=False, name='mlp'):
        super(MLP_model, self).__init__(name)
        self.X = inputs
        self.Y = outputs
        self.mlp = MLP(nUnits, activation=activation, X_center=X_center, regularization=regularization, reg_weight=reg_weight, positive_obs=positive_obs, name=name)
        self.link_parameter(self.mlp)

    def parameters_changed(self):
        Y_pred = self.mlp.predict(self.X)
        self.cost = np.sum((self.Y - Y_pred) ** 2)
        dL_dYpred = -(2 * Y_pred - 2 * self.Y)
        self.mlp.update_gradient(self.X, dL_dYpred)

    def predict(self, x):
        return self.mlp.predict(x)

    def log_likelihood(self):
        return -self.cost
