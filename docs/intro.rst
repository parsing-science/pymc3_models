Introduction to PyMC3 models
========================================

This library was inspired by my own work creating a re-usable Hierarchical Logistic Regression model.

To learn more, you can read this section, watch a 
`video from PyData NYC 2017 <https://www.youtube.com/watch?v=zGRnirbHWJ8>`__, or check out the 
`slides <https://github.com/parsing-science/pydata_nyc_nov_2017>`__ .

Quick intro to PyMC3
--------------------
When building a model with PyMC3, you will usually follow the same four steps:

- **Step 1: Set up** Parameterize your model, choose priors, and insert training data
- **Step 2: Inference** infer your parameters using MCMC sampling (e.g. NUTS) or variational inference (e.g. ADVI)
- **Step 3: Interpret** Check your parameter distributions and model fit
- **Step 4: Predict data** Create posterior samples with your inferred parameters

For a longer discussion of these steps, see :doc:`getting_started`.

Mapping between scikit-learn and PyMC3
--------------------------------------
This library builds a mapping between the steps above with the methods used by scikit-learn models.

+----------------+--------------------------------------+ 
| scikit-learn   | PyMC3                                | 
+================+======================================+
| Fit            | Step 1: Set up, Step 2: Inference    | 
+----------------+--------------------------------------+
| Predict        | Step 4: Predict Data                 | 
+----------------+--------------------------------------+
| Score          | Step 4: Predict data                 | 
+----------------+--------------------------------------+ 
| Save/Load      | ??                                   |
+----------------+--------------------------------------+
| ??             | Step 3: Interpret                    |
+----------------+--------------------------------------+

The question marks represent things that don't exist in the two libraries on their own. 


Comparing scitkit-learn, PyMC3, and PyMC3 Models
------------------------------------------------
Using the mapping above, this library creates easy to use PyMC3 models.

+----------------------------+-------------+-------------+--------------+
|                            |scikit-learn | PyMC3       | PyMC3 models | 
+============================+=============+=============+==============+
| Find model parameters      | Easy        | Medium      | Easy         |
+----------------------------+-------------+-------------+--------------+
| Predict new data           | Easy        | Difficult   | Easy         |
+----------------------------+-------------+-------------+--------------+
| Score a model              | Easy        | Difficult   | Easy         |
+----------------------------+-------------+-------------+--------------+
| Save a trained model       | Easy        | Impossible? | Easy         |
+----------------------------+-------------+-------------+--------------+
| Load a trained model       | Easy        | Impossible? | Easy         |
+----------------------------+-------------+-------------+--------------+
| Interpret Parameterization | N/A         | Easy        | Easy         |
+----------------------------+-------------+-------------+--------------+

