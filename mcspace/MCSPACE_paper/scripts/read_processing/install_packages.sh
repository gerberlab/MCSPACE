#!/bin/bash


# Function to install pip if not already installed
install_pip() {
    if ! command -v pip &> /dev/null; then
        echo "pip not found, installing pip..."
        sudo yum install -y python3-pip
    else
        echo "pip is already installed."
    fi
}

# Function to install system dependencies
install_dependencies() {
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y openssl-devel libffi-devel python3-devel
}

# Install pip
install_pip

# Install system dependencies
install_dependencies

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create and activate a new Conda environment (optional)
conda create -n myenv python=3.9 -y
source activate myenv


# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Clean pip cache
pip cache purge

# Install Cython
pip install cython

# Install packages from installed_packages.txt
conda install -c conda-forge -c bioconda seqkit vsearch gcc
pip install ultraplex==1.2.5
pip install pandas
sudo yum install pigz


#make usearch executable
chmod +x usearch10
