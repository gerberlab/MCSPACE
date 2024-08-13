from setuptools import setup 

setup(name='mcspace',
    version='0.1.0',
    description='MCSPACE model for MaPS-seq data',
    author='Gary Uppal',
    packages=['mcspace'],
    install_requires=[
                    'numpy==1.26.4',
                    'scikit-learn',
                    'matplotlib',
                    'seaborn',
                    'pandas',
                    'scipy',
                    'jupyterlab',
                    'ipykernel',
                    # 'scikit-bio',
                    'composition-stats',
                    'statsmodels'
                    ],
    python_requires=">=3.8",
)