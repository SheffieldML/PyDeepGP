# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""ARD scale inspection, pruning and restore for a :class:`deepgp.DeepGP`.

This recovers the capability of the Matlab ``hsvargplvmShowScales`` and
``PruneModel`` / ``RestorePrunedModel`` (CIP-0003, Phase 3), not their interface.

Scales are the ARD input sensitivities of each layer kernel, ``variance /
lengthscale**2`` for stationary kernels. The entry for layer ``i`` describes the
latent dimensions that feed that layer, so a near-zero entry marks a dimension
the layer does not use.

Pruning here *masks* irrelevant dimensions rather than resizing the model: the
kernel parameter that controls the dimension is driven to its "off" value
(very long lengthscale, or tiny variance for a linear kernel) and fixed, so the
optimiser cannot revive it. The latent array keeps its width. ``restore_model``
returns the original values exactly. Structural removal of columns, with the
speed-up it would bring, is not implemented.
"""
from collections import OrderedDict

import numpy as np

_OFF_LENGTHSCALE = 1e8
_OFF_VARIANCE = 1e-12


def _layer_kernels(model):
    """Ordered mapping from a label to the kernel of each layer (or MRD view)."""
    out = OrderedDict()
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'views'):
            for j, view in enumerate(layer.views):
                out['layer_%d/view_%d' % (i, j)] = view.kern
        else:
            out['layer_%d' % i] = layer.kern
    return out


def ard_scales(model, normalize=True):
    """Per-layer ARD relevance of each input dimension.

    Returns an ordered dict ``label -> 1-d array``. With ``normalize`` each array
    is divided by its maximum, so the most relevant dimension is 1.
    """
    out = OrderedDict()
    for label, kern in _layer_kernels(model).items():
        s = np.asarray(kern.input_sensitivity(summarize=True), dtype=float).copy()
        if normalize and s.max() > 0:
            s = s / s.max()
        out[label] = s
    return out


def show_scales(model, normalize=True, ax=None):
    """Bar chart of ``ard_scales``; returns the matplotlib figure. Needs matplotlib."""
    import matplotlib.pyplot as plt
    scales = ard_scales(model, normalize=normalize)
    if ax is None:
        fig, axes = plt.subplots(1, len(scales), squeeze=False,
                                 figsize=(3.2 * len(scales), 2.8))
        axes = axes[0]
    else:
        axes = np.atleast_1d(ax)
        fig = axes[0].figure
        if len(axes) != len(scales):
            raise ValueError('need %d axes, got %d' % (len(scales), len(axes)))
    for a, (label, s) in zip(axes, scales.items()):
        a.bar(np.arange(len(s)), s)
        a.set_title(label)
        a.set_xlabel('latent dimension')
    axes[0].set_ylabel('relative ARD scale' if normalize else 'ARD scale')
    fig.tight_layout()
    return fig


def _controlling_param(kern):
    """The ARD parameter whose value switches a dimension on or off, with its off value."""
    if hasattr(kern, 'lengthscale') and kern.lengthscale.size == kern.input_dim and kern.input_dim > 1:
        return kern.lengthscale, _OFF_LENGTHSCALE
    if hasattr(kern, 'variances') and kern.variances.size == kern.input_dim and kern.input_dim > 1:
        return kern.variances, _OFF_VARIANCE
    return None, None


def prune_model(model, threshold=1e-3):
    """Switch off input dimensions whose normalised ARD scale is below ``threshold``.

    At least one dimension per kernel is always kept. Layers whose kernel is not
    ARD (or has a single input) are skipped. Returns a dict ``label -> indices``
    of the pruned dimensions. Call ``restore_model`` to undo.
    """
    if getattr(model, '_pruned', None):
        raise RuntimeError('model is already pruned; call restore_model first')
    record = []
    report = OrderedDict()
    for label, kern in _layer_kernels(model).items():
        param, off = _controlling_param(kern)
        if param is None:
            report[label] = np.array([], dtype=int)
            continue
        s = np.asarray(kern.input_sensitivity(summarize=True), dtype=float)
        rel = s / s.max() if s.max() > 0 else np.ones_like(s)
        idx = np.where(rel < threshold)[0]
        if len(idx) == len(rel):
            idx = np.array([], dtype=int)
        report[label] = idx
        if len(idx) == 0:
            continue
        record.append((param, idx, param.values[idx].copy()))
        param[idx] = off
        for k in idx:
            param[k:k + 1].fix(warning=False)
    model._pruned = record
    return report


def restore_model(model):
    """Undo ``prune_model``, restoring the original parameter values exactly."""
    record = getattr(model, '_pruned', None)
    if record is None:
        raise RuntimeError('model has not been pruned')
    for param, idx, values in record:
        for k in idx:
            param[k:k + 1].unfix()
        param[idx] = values
    model._pruned = None
