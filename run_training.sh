#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Create a conda environment named Mesho_env with Python 3.10
echo "Creating conda environment 'Mesho_env' with Python 3.10..."
conda create -y -n Mesho_env python=3.10

# Activate the environment
echo "Activating environment 'Mesho_env'..."
source activate Mesho_env

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages from requirements.txt
echo "Installing required packages from 'requirements.txt'..."
pip install -r requirements.txt

# Verify that TensorFlow detects the GPUs
echo "Verifying TensorFlow GPU setup..."
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Start TensorBoard in the background
echo "Starting TensorBoard..."
tensorboard --logdir=logs/ --port=6006 &

# Run the training script
echo "Starting model training..."
python model_training.py

# Deactivate the conda environment after training is complete
echo "Deactivating environment 'Mesho_env'..."
conda deactivate

echo "Training complete."