"""Testing for the custom kernel for Gaussian processes."""

import pytest
import numpy as np
from scipy.spatial import distance
from sklearn.gaussian_process.covar_functions import  *
from sklearn.gaussian_process.kernels import Sum, Exponentiation, Product, CustomKernel, RBF, ConstantKernel, DotProduct, generate_kernel
from sklearn.gaussian_process._gpr import GaussianProcessRegressor


# test data
X_train = [[i] for i in range(100)]
Y_train = [x*np.sin(x) for x in range(100)]
X_test = [[i*np.pi] for i in range(10,20)]


""" Tests for covariance functions """

def test_kernel_no_params():
    # test a kernel with no parameters
    def cov(x,y):
        return np.exp( - np.linalg.norm(x-y))

    kernel = generate_kernel(cov)

    # predict with a model using the above kernel
    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)

    assert(len(preds) == 10)

def test_constant():
    # define a custom kernel with the constant covariance function, and set hyperparameters
    kernel = generate_kernel(constant, constant_grad, const =1, const_bounds = (1e-5, 1e5))

    # create a GPR with the above kernel and use it on the data
    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds1 = model.predict(X_test)

    # do the same with a ConstantKernel, with the same parameters
    kernel = ConstantKernel()

    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds2 = model.predict(X_test)

    # assert that the two give almost the same predictions
    assert (np.all(np.isclose(preds1, preds2)))


def test_rbf():

    # define a custom kernel with the RBF covariance function, and set hyperparameters
    kernel = generate_kernel(rbf, rbf_grad, length_scale = 1, length_scale_bounds = (1e-5, 1e5))

    # create a GPR with the above kernel and use it on the data
    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds1 = model.predict(X_test)

    # do the same with an RBF, with the same length_scale and bounds
    kernel = RBF(length_scale = 1)

    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds2 = model.predict(X_test)

    # assert that the two give almost the same predictions
    assert(np.all(np.isclose(preds1, preds2)))

def test_dot_prod():

    # define a custom kernel with the dot product covariance function, and set hyperparameters
    kernel = generate_kernel(dot_prod, dot_prod_grad, sigma = 1, sigma_bounds = (1e-5, 1e5))

    # create a GPR with the above kernel and use it on the data
    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds1 = model.predict(X_test)

    # do the same with an RBF, with the same length_scale and bounds
    kernel = DotProduct()

    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds2 = model.predict(X_test)

    # assert that the two give almost the same predictions
    assert(np.all(np.isclose(preds1, preds2)))

def test_sum():

    # define rbf and dot product custom kernels, and set hyperparameters
    custom_rbf_kernel = generate_kernel(rbf, rbf_grad, length_scale = 1, length_scale_bounds = (1e-5, 1e5))
    custom_dot_prod_kernel = generate_kernel(dot_prod, dot_prod_grad, sigma = 1, sigma_bounds = (1e-5, 1e5))

    sum_custom_kernel = Sum(custom_dot_prod_kernel, custom_rbf_kernel)
    model = GaussianProcessRegressor(kernel=sum_custom_kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds1 = model.predict(X_test)

    sum_kernel = Sum(DotProduct(), RBF())
    model = GaussianProcessRegressor(kernel=sum_kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds2 = model.predict(X_test)

    assert(np.all(np.isclose(preds1, preds2)))

def test_product():

    # define rbf and dot product custom kernels, and set hyperparameters
    custom_rbf_kernel = generate_kernel(rbf, rbf_grad, length_scale = 1, length_scale_bounds = (1e-5, 1e5))
    custom_dot_prod_kernel = generate_kernel(dot_prod, dot_prod_grad, sigma = 1, sigma_bounds = (1e-5, 1e5))

    prod_custom_kernel = Product(custom_dot_prod_kernel, custom_rbf_kernel)
    model = GaussianProcessRegressor(kernel=prod_custom_kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds1 = model.predict(X_test)

    prod_kernel = Product(DotProduct(), RBF())
    model = GaussianProcessRegressor(kernel=prod_kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds2 = model.predict(X_test)

    assert(np.all(np.isclose(preds1, preds2)))


def test_exponentiation():

    # define an rbf custom kernel, and set hyperparameters
    custom_rbf_kernel = generate_kernel(rbf, rbf_grad, length_scale = 1, length_scale_bounds = (1e-5, 1e5))

    exp_custom_kernel = Exponentiation(custom_rbf_kernel, 5)
    model = GaussianProcessRegressor(kernel=exp_custom_kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds1 = model.predict(X_test)

    exp_kernel = Exponentiation(RBF(), 5)
    model = GaussianProcessRegressor(kernel=exp_kernel, random_state=0)
    model.fit(X_train, Y_train)
    preds2 = model.predict(X_test)

    assert(np.all(np.isclose(preds1, preds2)))