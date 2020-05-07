#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:16:56 2019

@author: maxime
Tuto utilisé : http://sametmax.com/creer-un-setup-py-et-mettre-sa-bibliotheque-python-en-ligne-sur-pypi/
"""

from setuptools import setup, find_packages

with open('rl_baselines/__init__.py') as f:
    for line in f:
        if '__version__' in line:
            exec(line)

setup(
    name='rl_baselines',
    version=__version__,
    packages=find_packages(),

    author="Uncharteam",
    author_email="uncharteam@unchartech.com",
    description="Reinforcement-Learning baselines.",
    long_description=open('README.md').read(),

    install_requires=[
        'atari-py>=0.1.7',
        'opencv-python',
        'gym[atari,box2d]>=0.10',
        'matplotlib>=3.0',
        'torch>=1.0',
        'clize',
    ],

    # Active la prise en compte du fichier MANIFEST.in
    include_package_data=True,

    url='https://github.com/unchartech/rl_baselines',

    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
