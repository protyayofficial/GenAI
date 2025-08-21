# VAE: Variational Autoencoder

This folder contains an implementation of a Variational Autoencoder (VAE) for MNIST and FashionMNIST datasets. The VAE is a generative model that learns to encode data into a latent space and reconstruct it, enabling generation of new samples. You can create custom data loaders and train the VAE model on other datasets and generate the results. 

## Features

- Train and test VAE on MNIST or FashionMNIST.
- Visualize reconstructions and generated samples.
- Save model checkpoints and results.
- Modular code for easy extension.

## Usage

### 1. Install dependencies

Make sure you have installed the requirements from the main repository:

```sh
pip install -r ../requirements.txt
```

### 2. Training

Train the VAE model:

```sh
python train.py --dataset mnist --epochs 20 --batch_size 128
```

You can change `--dataset` to `fashionmnist` and adjust other parameters as needed.

### 3. Testing

Evaluate the trained model:

```sh
python test.py --dataset mnist --checkpoint vae_mnist.pth
```

### 4. Visualization

Generated samples and reconstructions are saved in the `vae_results/` folder.

## Scripts

- `train.py`: Training script for VAE.
- `test.py`: Testing and evaluation script.
- `utils.py`: Helper functions for data loading and visualization.
- `script.sh`: Example bash script for running experiments.

## Configuration

You can customize hyperparameters and dataset options via command-line arguments or by editing `script.sh`.

## References

- [Kingma & Welling, Auto-Encoding Variational Bayes (2013)](https://arxiv.org/abs/1312.6114)

---

*For questions or contributions, please open an issue or pull request in the main repository.*