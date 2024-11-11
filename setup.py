from setuptools import setup 

setup(name='mcspace',
    version='1.0.0',
    description='MCSPACE model for analyzing MaPS-seq and SAMPL-seq co-localization data',
    author='Gary Uppal',
    packages=['mcspace'],
    install_requires=[
                    'numpy==1.26.4',
                    'scikit-learn',
                    'matplotlib',
                    'seaborn',
                    'pandas',
                    'scipy',
                    'scikit-learn',
                    'jupyterlab',
                    'ipykernel',
                    'biopython',
                    'ete3',
                    'networkx',
                    'composition-stats',
                    'statsmodels'
                    ],
    entry_points={
        'console_scripts': [
            'mcspace=mcspace.cli:main'
        ]
    },
    python_requires=">=3.8",
)