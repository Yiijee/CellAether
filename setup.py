from setuptools import setup, find_packages

setup(
    name='CellAether',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'PyQt6',
        'tifffile',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'CellAether = CellAether.main_run:main',
        ],
    },
)
