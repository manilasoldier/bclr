from setuptools import setup 

VERSION = '0.2.0'
DESCRIPTION = 'bclr - Bayesian changepoint detection via Logistic Regression'

setup(
    name="bclr",
    version=VERSION,
    author="Andrew M. Thomas and Michael Jauch",
    maintainer="Andrew M. Thomas",
    package=['bclr'],
    install_requires=[
	'matplotlib',
	'pandas',
	'numpy',
	'detectda >= 0.4.5',
	'joblib',
	'tabulate',
	'scikit-learn >= 1.3.0',
	'scipy',
	'polyagamma >= 1.3.5.',
    ],
    classifiers = [
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python'
    ]
)
