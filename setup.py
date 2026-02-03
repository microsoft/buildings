#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Akram Zaytar, Tammy Glazer, Gilles Quentin Hacheme, Caleb Robinson",
    author_email='akramzaytar@microsoft.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Contains guidelines for data preparation, deep learning training, and inference methods to produce TEMPO (raw) building density and height maps.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='tempo',
    name='tempo',
    packages=find_packages(include=['tempo', 'tempo.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/microsoft/buildings',
    version='0.1.0',
    zip_safe=False,
)
