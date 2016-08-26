# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


def gen_timestring(prjname=None):
    from datetime import datetime
    timenow = datetime.now()
    timestr = timenow.strftime('%Y:%m:%d_%H:%M:%S')
    if prjname is None:
        return timestr
    else:
        return prjname+'_'+timestr
    
def comp_mapping(X, Y):
    from GPy.core.parameterization.variational import VariationalPosterior
    X = X.mean.values if isinstance(X, VariationalPosterior) else X
    Y = Y.mean.values if isinstance(Y, VariationalPosterior) else Y
    from scipy.linalg import lstsq
    W = lstsq(X,Y)[0]
    return W
