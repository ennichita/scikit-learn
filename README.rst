.. -*- mode: rst -*-

|Azure|_ |Travis|_ |Codecov|_ |CircleCI|_ |PythonVersion|_ |PyPi|_ |DOI|_

.. |Azure| image:: https://dev.azure.com/scikit-learn/scikit-learn/_apis/build/status/scikit-learn.scikit-learn?branchName=master
.. _Azure: https://dev.azure.com/scikit-learn/scikit-learn/_build/latest?definitionId=1&branchName=master

.. |Travis| image:: https://api.travis-ci.org/scikit-learn/scikit-learn.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn/scikit-learn

.. |Codecov| image:: https://codecov.io/github/scikit-learn/scikit-learn/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/github/scikit-learn/scikit-learn?branch=master

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn/scikit-learn/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn/scikit-learn

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/scikit-learn.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/scikit-learn.svg

.. |PyPi| image:: https://badge.fury.io/py/scikit-learn.svg
.. _PyPi: https://badge.fury.io/py/scikit-learn

.. |DOI| image:: https://zenodo.org/badge/21369/scikit-learn/scikit-learn.svg
.. _DOI: https://zenodo.org/badge/latestdoi/21369/scikit-learn/scikit-learn

scikit-learn
============

scikit-learn is a Python module for machine learning built on top of
SciPy and is distributed under the 3-Clause BSD license.

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <http://scikit-learn.org/dev/about.html#authors>`__ page
for a list of core contributors.

It is currently maintained by a team of volunteers.

Website: http://scikit-learn.org


Installation
------------

Dependencies
~~~~~~~~~~~~

scikit-learn requires:

- Python (>= 3.5)
- NumPy (>= 1.11.0)
- SciPy (>= 0.17.0)
- joblib (>= 0.11)

**Scikit-learn 0.20 was the last version to support Python 2.7 and Python 3.4.**
scikit-learn 0.21 and later require Python 3.5 or newer.

Scikit-learn plotting capabilities (i.e., functions start with ``plot_``
and classes end with "Display") require Matplotlib (>= 1.5.1). For running the
examples Matplotlib >= 1.5.1 is required. A few examples require
scikit-image >= 0.12.3, a few examples require pandas >= 0.18.0.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of numpy and scipy,
the easiest way to install scikit-learn is using ``pip``   ::

    pip install -U scikit-learn

or ``conda``::

    conda install scikit-learn

The documentation includes more detailed `installation instructions <http://scikit-learn.org/stable/install.html>`_.


Changelog
---------

See the `changelog <http://scikit-learn.org/dev/whats_new.html>`__
for a history of notable changes to scikit-learn.

Development
-----------

We welcome new contributors of all experience levels. The scikit-learn
community goals are to be helpful, welcoming, and effective. The
`Development Guide <http://scikit-learn.org/stable/developers/index.html>`_
has detailed information about contributing code, documentation, tests, and
more. We've included some basic information in this README.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/scikit-learn/scikit-learn
- Download releases: https://pypi.org/project/scikit-learn/
- Issue tracker: https://github.com/scikit-learn/scikit-learn/issues

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/scikit-learn/scikit-learn.git

Contributing
~~~~~~~~~~~~

To learn more about making a contribution to scikit-learn, please see our
`Contributing guide
<https://scikit-learn.org/dev/developers/contributing.html>`_.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the
source directory (you will need to have ``pytest`` >= 3.3.0 installed)::

    pytest sklearn

See the web page http://scikit-learn.org/dev/developers/advanced_installation.html#testing
for more information.

    Random number generation can be controlled during testing by setting
    the ``SKLEARN_SEED`` environment variable.

Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

Before opening a Pull Request, have a look at the
full Contributing page to make sure your code complies
with our guidelines: http://scikit-learn.org/stable/developers/index.html


Project History
---------------

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <http://scikit-learn.org/dev/about.html#authors>`__ page
for a list of core contributors.

The project is currently maintained by a team of volunteers.

**Note**: `scikit-learn` was previously referred to as `scikits.learn`.


Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- HTML documentation (stable release): http://scikit-learn.org
- HTML documentation (development version): http://scikit-learn.org/dev/
- FAQ: http://scikit-learn.org/stable/faq.html

Communication
~~~~~~~~~~~~~

- Mailing list: https://mail.python.org/mailman/listinfo/scikit-learn
- IRC channel: ``#scikit-learn`` at ``webchat.freenode.net``
- Stack Overflow: https://stackoverflow.com/questions/tagged/scikit-learn
- Website: http://scikit-learn.org

Citation
~~~~~~~~

If you use scikit-learn in a scientific publication, we would appreciate citations: http://scikit-learn.org/stable/about.html#citing-scikit-learn

Fork details
~~~~~~~~~~~~

To use a Gaussian Process Regressor (GPR), we need to define a Kernel function.
This kernel function can be thought of as a similarity measure between two input data points.
Any inner product can be used as a kernel function, as well as the sum,
product or exponentiation of already existing kernels. This generation process gives a huge range of kernels,
of which only about 10 are coded in the scikit library.
What I’ve written allows for a quick definition of the kernel function (ex. f(x,y) = a*e^(-norm(x-y)),
the CustomKernel class taking care of the actual Kernel object generation.

Most kernels commonly used are actually isotropic. This means that they only depend on the distance between the two input points.
Another issue with the scikit library is that the distance metric are hard coded into the kernel objects.
For example, the RBF object uses the squared euclidean (L2) distance, which has two drawbacks:

- We can’t use features  like fingerprints, graphs, etc. as they’re not compatible with the distance metric
- We’d have to write a kernel object for each new metric tried, which is very tedious

One additional complication is that most kernels have a tunable hyperparameter. This is used by the GPR, when fitting the train data:

- It used the kernel with some default hyperparameter value on the data
- Computes the log-marginal likelihood for this default value
- Performs gradient descent to tune the hyperparameter and optimize the likelihood
- Saves the hyperparameter value and uses it for predicting future data

One might notice that to perform gradient descent, we need to have a gradient function.
This is, again, hard coded into the existing Kernel objects in scikit.
For custom kernels and metrics, the only workaround is the PairwiseKernel object, which approximates it.
The drawbacks for this are:

- It can only use a single hyperparameter galled ‘gamma’
- The gradient is numerically computed, which is both slow and innacurate
- There is still no separation between the metric distance and the kernel function. That is, for different metrics (for different features perhaps), you need to code different functions.

The code I’ve written also allows for a gradient function definition, which is used by the optimizer when fitting the data.

Usage
~~~~~
