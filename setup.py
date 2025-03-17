#!/usr/bin/env python

from setuptools import setup

requires = ['matplotlib',
            'numpy',
            ]

setup(name='compton_simulator',
      version='0.1',
      install_requires=requires,
      description=('Simulación de un experimento de'
                   'dispersión Compton en correlaciones'),
      author="Mauricio Matera",
      author_email='matera@fisica.unlp.edu.ar',
      packages=find_packages(),
      )
