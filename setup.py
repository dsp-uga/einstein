"""A setuptools based setup module.

References:
(a) https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
import os


repo_dir = os.path.abspath(__file__)
# Long description is just the contents of README.md
long_description = 'Read README.md for long description'

setup(
	# Users can install the project with the following command:
	#		$ pip install einstein
	#
	# It will live on PyPi at:
	#		https://pypi.org/project/einstein/
	name='einstein',
	# Versions should coply with PEP 440 : 
	# 		https://www.python.org/dev/peps/pep-0440/
	version='0.0.1-dev',
	# Packages can be manually mentioned, or `setuptools.find_packages`
	# can be used for this purpose.
	packages=find_packages(exclude=["*test_*.*", "*_test.*", "tests"]),
	description='Solar Irradiance Prediction using PySpark',
	long_description=long_description,
	# Corresponds to the Home Page of the metadata field
	url='https://github.com/dsp-uga/einstein',
	# Name and email addresses of project owners.
	author='Aashish Yadavally, Anirudh Maurya, Jayant Parashar',
	author_email='aashish.yadavally1995@gmail.com',
	classifiers=[
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Atmospheric Science',

		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
	],
	python_requires='>=3.6',
	)
