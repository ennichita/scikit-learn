import pytest
import numpy as np
from scipy.spatial import distance
from sklearn.gaussian_process.covar_functions import  *
from sklearn.gaussian_process.kernels import ExpSineSquared, Sum, Exponentiation, Product, CustomKernel, RBF, ConstantKernel, DotProduct, generate_kernel
from sklearn.gaussian_process._gpr import GaussianProcessRegressor

X_train = [[i] for i in range(100)]
Y_train = [x*np.sin(x) for x in range(100)]
X_test = [[i*np.pi] for i in range(10,20)]

def test_sandbox():

    kernel = generate_kernel(rbf, rbf_grad)
    model = GaussianProcessRegressor(kernel = kernel, random_state = 0)
    model.fit(X_train, Y_train)
    preds1 = model.predict(X_test)
