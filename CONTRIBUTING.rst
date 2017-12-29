Introduction
================

Thank you for considering contributing to PyMC3 Models! This project is intended to be a space where anyone can share models they've built.

Please read these guidelines before submitting anything to the project. As of the first release, I'm the only person working on this project so respecting these guidelines will help me get back to you more quickly.

Some ways to contribute:

- Open an issue on the `Github Issue Tracker <https://github.com/parsing-science/pymc3_models/issues>`__. (Please check that it has not already been reported or addressed in a PR.)
- Improve the docs!
- Add a new model. Please follow the guidelines below.
- Add/change existing functionality in the base model class
- Something I haven't thought of?
  
Pull Requests
------------------
To create a PR against this library, please fork the project and work from there.

Steps:

1. Fork the project via the Fork button on Github

2. Clone the repo to your local disk.

3. Create a new branch for your PR.
::

    git checkout -b my-awesome-new-feature

4. Install requirements (probably in a virtual environment)
::

    virtualenv venv
    source venv/bin/activate
    pip install -r requirements-dev.txt
    pip install -r requirements.txt

5. Develop your feature
   
6. Submit a PR!
   
PR Checklist
=============

- Ensure your code has followed the Style Guidelines below
- Make sure you have written unittests where appropriate
- Make sure the unittests pass
::

    source venv/bin/activate
    python -m unittest discover -cv

- Update the docs where appropriate. You can rebuild them with the commands below.
::

    cd pymc3_models/docs
    sphinx-apidoc -f -o api/ ../pymc3_models/
    make html

Notes for new models
=====================

- New models should be put into the models directory. 
- Make the file name the same as the class name; be explicit, e.g. HierarchicalLogisticRegression, not HLR.
- Try to write some simple unittests for your model. I do not recommend using NUTS in your unittests if you have a complex model because the tests will take hours to run.
- [Optional] Please create a Jupyter notebook in the notebooks folder with the same name as your model class. In it, show a simple example of how to use your model. Synthetic data is fine to use.

Style Guidelines
=================
For the most part, this library follows PEP8 with a couple of exceptions. 

Notes:

- Indent with 4 spaces
- Lines can be 110 characters long
- Docstrings should be written as numpy docstrings
- Your code should be Python 3 compatible
- When in doubt, follow the style of the existing code
