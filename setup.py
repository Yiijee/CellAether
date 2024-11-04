from setuptools import setup, find_packages

setup(
    name='CellAether',
    version='0.2',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'CellAether': ['kmeans.pkl'],
    },
    install_requires=[
        'PyQt6',
        'tifffile',
        'numpy',
        'pandas',
        'scikit-learn',
        'spotiflow'

    ],
    entry_points={
        'console_scripts': [
            'CellAether = CellAether.main_run:main',
        ],
    },
)
