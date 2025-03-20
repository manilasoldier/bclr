from setuptools import setup, find_packages 
import codecs

DESCRIPTION = 'bclr - Bayesian changepoint detection via Logistic Regression'
with codecs.open("DESCRIPTION.md", encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_TYPE = "text/markdown"
VERSION = {}
with open("bclr/_version.py") as fp:
    exec(fp.read(), VERSION)

setup(
    name="bclr",
    version=VERSION['__version__'],
    author="Andrew M. Thomas and Michael Jauch",
    author_email = "<me@andrewmthomas.com>",
    maintainer="Andrew M. Thomas",
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    long_description_content_type = LONG_DESCRIPTION_TYPE,
    packages=find_packages(),
    include_package_data=True,
    package_data={'bclr': ['bclr/tests/*.npy']},
    install_requires=[
	'wheel',
	'matplotlib',
	'pandas',
	'numpy <= 1.26.4',
	'detectda >= 0.5.3',
	'joblib',
	'tabulate',
	'scikit-learn >= 1.3.0',
	'scipy',
	'polyagamma >= 1.3.6',
        'ruptures >= 1.1.8'
    ],
    keywords = ['changepoint', 'bayesian', 'logistic regression'],
    classifiers = [
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python'
    ]
)
