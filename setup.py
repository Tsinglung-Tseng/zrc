from setuptools import setup

setup(
    name='zrc',
    version='0.0.1',
    package_dir={'': 'src/python'},
    install_requires=[
        'numpy',
        'pandas',
        'plotly==4.9.0',
    ],
)
