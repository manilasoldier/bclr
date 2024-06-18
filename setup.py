from setuptools import setup, find_packages 

DESCRIPTION = 'bclr - Bayesian changepoint detection via Logistic Regression'
VERSION = {}
with open("bclr/_version.py") as fp:
    exec(fp.read(), VERSION)

setup(
    name="bclr",
    version=VERSION['__version__'],
    author="Andrew M. Thomas and Michael Jauch",
    maintainer="Andrew M. Thomas",
    packages=find_packages(),
    install_requires=[
	'matplotlib',
	'pandas',
	'numpy',
	'detectda >= 0.4.5',
	'joblib',
	'tabulate',
	'scikit-learn >= 1.3.0',
	'scipy',
	'polyagamma >= 1.3.5',
    'ruptures >= 1.1.8'
    ],
    classifiers = [
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python'
    ]
)
