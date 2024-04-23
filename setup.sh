#!/bin/sh

# Conda Version
conda -V 

# List Existing Conda Environments in System
conda info --envs

# Update Conda
conda update conda

# Upgrade PIP
pip install --upgrade pip

# Remove Existing Old Env
conda env remove -n data_science_env

# Create Conda Environment
conda create --name data_science_env python=3.12 -y

# Activate Conda Environment
conda activate data_science_env

# Install Python Packages
pip install -r requirements.txt

# Uninstall Python Packages
pip uninstall -r requirements.txt

# Check Python Libraries
conda list 
pip freeze

# Conda Env Export 
conda env export > data_science_env.yml

# Python Version
python --version

# Python Path
which python

# Deactivate Conda Environment
conda deactivate