from setuptools import setup, find_packages

with open('README.rst') as r:
    readme = r.read()

with open('AUTHORS.txt') as a:
    # reSt-ify the authors list
    authors = ''
    for author in a.read().split('\n'):
        authors += '| '+author+'\n'

with open('LICENSE') as l:
    license = l.read()


setup(
    name='pymc3_models',
    version='1.1.2',
    description='Custom PyMC3 models built on top of the scikit-learn API',
    long_description=readme,
    author='Nicole Carlson',
    author_email='nicole@parsingscience.com',
    url='https://github.com/parsing-science/pymc3_models',
    license=license,
    packages=find_packages(),
    package_data={'docs': ['*']},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'joblib',
        'matplotlib',
        'numpy',
        'pandas>=0.19',
        'pymc3>=3.3',
        'scipy',
        'seaborn',
        'sklearn'
    ],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4'
    ]
)
