#!/bin/bash
# Script to train and test VAE on MNIST

# Activate your conda/venv environment if needed
# source ~/miniconda3/bin/activate myenv

# Train the VAE
echo "Starting training..."
python train.py --dataset MNIST --epochs 30 --latent_dim 32

# Test the trained VAE
echo "Starting testing..."
python test.py --dataset MNIST --checkpoint vae_mnist.pth --latent_dim 32

echo "Done."
