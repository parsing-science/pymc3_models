PyMC3 Models
================

Custom PyMC3 models built on top of the scikit-learn API. Check out the `docs <http://pymc3-models.readthedocs.io/>`__.

Features
------------------

- Reusable PyMC3 models including LinearRegression and HierarchicalLogisticRegression
- A base class, BayesianModel, for building your own PyMC3 models

Installation
------------------
The latest release of PyMC3 Models can be installed from PyPI using ``pip``:

::

    pip install pymc3_models

The current development branch of PyMC3 Models can be installed from GitHub, also using ``pip``:

::

    pip install git+https://github.com/parsing-science/pymc3_models.git

To run the package locally (in a virtual environment):

::

    git clone https://github.com/parsing-science/pymc3_models.git
    cd pymc3_models
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt

Usage
------------------
Since PyMC3 Models is built on top of scikit-learn, you can use the same methods as with a scikit-learn model.

::

    from pymc3_models import LinearRegression

    LR = LinearRegression()
    LR.fit(X, Y)
    LR.predict(X)
    LR.score(X, Y)


Contribute
------------------
For more info, see `CONTRIBUTING <https://github.com/parsing-science/pymc3_models/blob/master/CONTRIBUTING.rst>`__.

Contributor Code of Conduct
------------------------------------
Please note that this project is released with a `Contributor Code of Conduct <http://contributor-covenant.org/>`__. By participating in this project you agree to abide by its terms. See `CODE_OF_CONDUCT <https://github.com/parsing-science/pymc3_models/blob/master/CODE_OF_CONDUCT.rst>`__.

Acknowledgments
------------------
This library is built on top of `PyMC3 <http://docs.pymc.io/>`__ and `scikit-learn <http://scikit-learn.org>`__.

License
------------------
`Apache License, Version 2.0 <https://github.com/parsing-science/pymc3_models/blob/master/LICENSE>`__
