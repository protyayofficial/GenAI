# GenAI: Generative Models Playground

This repository is a modular playground for experimenting with modern generative models in deep learning. It is designed for research, prototyping, and educational purposes, with a focus on extensibility and reproducibility.

## Structure

- **VAE**: Variational Autoencoder for MNIST and FashionMNIST, including training, testing, and visualization scripts.
- *(Coming Soon)* **DDPM, DDIM, etc.**

## Features

- Modular code for easy extension and integration of new models.
- Scripts for training, testing, and visualizing results.
- Support for multiple datasets.
- Reproducible experiments with clear configuration options.
- Results and checkpoints are saved for analysis and comparison.

## Getting Started

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/GenAI.git
    cd GenAI
    ```

2. **Set up your environment:**
    - Python 3.11+
    - Recommended: [conda](https://docs.conda.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html)
    - Install dependencies:
        ```sh
        pip install -r requirements.txt
        ```

3. **Run VAE experiments:**
    ```sh
    cd VAE
    bash script.sh
    ```

    - Training and testing parameters can be customized in [script.sh](http://_vscodecontentref_/0) or via CLI arguments in [train.py](http://_vscodecontentref_/1) and [test.py](http://_vscodecontentref_/2).

## Results

Generated samples, reconstructions, and analysis plots are saved in each model's `*_results/` folder.

## Contributing

Pull requests and issues are welcome! Please open an issue to discuss major changes or new model integrations.

## License

This project is licensed under the MIT License.

---

*This repository is under active development. More generative models and guidance techniques will be added soon.*