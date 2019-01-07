from setuptools import setup, find_packages

with open('AUTHORS.txt') as a:
    # reSt-ify the authors list
    authors = ''
    for author in a.read().split('\n'):
        authors += '| '+author+'\n'

with open('pymc3_models/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()


setup(
    name='pymc3_models',
    version=__version__,
    description='Custom PyMC3 models built on top of the scikit-learn API',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Nicole Carlson',
    author_email='nicole@parsingscience.com',
    url='https://github.com/parsing-science/pymc3_models',
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
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4'
    ]
)
