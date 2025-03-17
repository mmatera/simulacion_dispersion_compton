#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='compton_simulator',
    version='0.1',
    install_requires=[
        'matplotlib',
        'numpy',
    ],
    description=('Simulación de un experimento de'
                 'dispersión Compton en correlaciones'),
    author="Mauricio Matera",
    author_email='matera@fisica.unlp.edu.ar',
    packages=find_packages(),  # Ensures 'compton_simulator' is included
)

