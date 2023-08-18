from setuptools import setup 

setup(name='mcspace',
    version='0.1.0',
    description='MC-SPACE model for MaPS-seq data',
    author='Gary Uppal',
    packages=['mcspace'],
    install_requires=[
                    'numpy',
                    'scikit-learn',
                    'matplotlib',
                    'seaborn',
                    'pandas',
                    'scipy',
                    'jupyterlab',
                    'ipykernel',
                    ],
    python_requires=">=3.8",
)