================
PyMC3 Models
================

Custom PyMC3 models built on top of the scikit-learn API.

Features
========

Installation
========
The latest release of PyMC3 Models can be installed from PyPI using ``pip``:

::

    pip install pymc3_models

The current development branch of PyMC3 Models can be installed from GitHub, also using ``pip``:

::

    pip install git+https://github.com/parsing-science/pymc3_models.git

Usage
========
Using PyMC3 Models is as simple as using scikit-learn.

::
    from pymc3_models import LinearRegression
    LR = LinearRegression()
    LR.fit(X, Y)
    LR.predict(X)
    LR.score(X, Y)


Contribute
========
Please contribute to this project. For more info, see `CONTRIBUTE <https://github.com/parsing_science/pymc3_models/blob/master/CONTRIBUTE.rst>`.

Contributor Code of Conduct
========
Please note that this project is released with a [Contributor Code of
Conduct](http://contributor-covenant.org/). By participating in this project
you agree to abide by its terms. See `CODE_OF_CONDUCT <https://github.com/parsing_science/pymc3_models/blob/master/CODE_OF_CONDUCT.rst>`.

License
========
`Apache License, Version
2.0 <https://github.com/parsing_science/pymc3_models/blob/master/LICENSE>`__