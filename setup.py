#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
This work is based on the paper: http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=32278
A Tree-Based Statistical Language Model for Natural Language Speech Recognition (1989) by L R Bahl, P F Brown, P V de Souza, R L Mercer
Python is the primary language to be used in this implementation.
Read the documentation: https://github.com/iisc-sa-open/trsl/wiki to get started.
"""

from setuptools import setup
setup(
    name='trsl',
    version='1.0',
    url='https://github.com/iisc-open-sa/trsl',
    license='MIT',
    author="Indian Institute of Science's Speech and Audio group",
    author_email='https://github.com/iisc-sa-open/trsl/issues',
    description='Python implementation of the A Tree-Based Statistical Language Model for Natural Language Speech Recognition paper',
    long_description=__doc__,
    packages=['trsl', 'trsl.sets'],
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
