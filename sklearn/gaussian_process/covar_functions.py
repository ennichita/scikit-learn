""" Covariance function file """
import numpy as np
from scipy.spatial import distance

class CovarianceFunction:
    """
    Class wrapping up a covariance function used within a Custom Kernel

    Parameters
    ----------
    function : function object
        A covariance function of the shape (x, y, *params) -> float
    grad_function: function object (opt)
        A function computing the gradient with respect to a given
        valid parameter, on a pair of points x, y.
    metric: fuction object (optional)
        Metric function used to define the distance between x and y.

    #TODO
    Examples
    --------
    """

    def __init__(self, function, grad_function=None, metric = None):
        self.metric = metric
        self.function = function
        self.grad_function = grad_function

    def __eq__(self, other):
        return self.function == other.function and \
               self.grad_function == other.grad_function and \
               self.metric == other.metric

    # Compute the distance between two points
    def __call__(self, x, y, params={}):

        if self.metric is None:
            return self.function(x, y, **params)
        else:
            return self.function(x, y, metric = self.metric, **params)

    # Compute the gradient of the specified parameter as given by the grad function
    # Note: The parameters are first log-transformed before input to optimizers,
    # so the grad_function must compute the derivative of the metric with respect to
    #  the log of the parameter
    def grad(self, x, y, params, param=None):

        if self.metric is None:
            return self.grad_function(x, y, param=None, **params)
        else:
            return self.grad_function(x, y, param = None, metric = self.metric, **params)


""" 
    Covariance functions and corresponding gradients 
    
    Functions and corresponding gradients must be defined as follows:
    
    def function(x, y, param_1, param_2 ... param_N):
        ***
    
    def function_grad(x, y, param_1, param_2, ... param_N, param = None):
        ***,   
        where param is a string in the set {'param_1', 'param_2', ... 'param_N'}
        and it signifies the parameter to use for computing the gradient 
"""

# constant function
def constant(x, y, const):
    return const
# constant function gradient
def constant_grad(x, y, const, param=None):
    if param is None:
        param = 'const'

    if param == 'const':
        return const

    else:
        raise Warning(f"Parameter {param} not defined for constant_grad")

# RBF function
def rbf(x, y, length_scale, metric = distance.sqeuclidean):
    return np.exp(-1 / 2 * metric(x, y) / length_scale ** 2)
# RBF function gradient
def rbf_grad(x, y, length_scale, metric = distance.sqeuclidean, param=None):
    if param is None:
        param = 'length_scale'

    if param == 'length_scale':
        delta = metric(x,y)
        result = np.exp(-1 / 2 * delta / length_scale ** 2) * delta / (length_scale ** 2)
        return result

    else:
        raise Warning(f"Parameter {param} not defined for rbf_grad")

# dot product function
def dot_prod(x, y, sigma):
    return sigma**2 + np.dot(x,y)
# dot product function gradient
def dot_prod_grad(x, y, sigma, param = None):
    if param is None:
        param = "sigma"

    if param == "sigma":
        return 2*sigma**2

    else:
        raise Warning(f"Parameter {param} not defined for dot_prod_grad")

# exponential sin squared function
def exp_sin_sq(x, y, length_scale, periodicity, metric=distance.sqeuclidean):
    return np.exp(-2 * (np.sin(np.pi / periodicity * metric(x, y))/length_scale)**2)
# dot product function gradient
def exp_sin_sq_grad(x, y, length_scale, periodicity, metric = distance.sqeuclidean, param = None):
    if param is None:
        param = "length_scale"

    # TODO compute the gradients
    if param == "length_scale":
        return 1

    if param == "periodicity":
        return 1

    else:
        raise Warning(f"Parameter {param} not defined for exp_sin_sq_grad")


