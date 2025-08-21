import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os

# ---------- Data Loader ----------
def get_dataloaders(dataset_name="MNIST", batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name.upper() == "MNIST":
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif dataset_name.upper() == "FASHIONMNIST":
        train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ---------- Loss ----------
def loss_function(recon_x, x, mu, logvar, return_parts=False):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    if return_parts:
        return BCE, KLD
    return BCE + KLD

def ensure_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# ---------- Visualizations ----------
def show_reconstructions(model, data_loader, device="cuda", n=8, save_path=None):
    model.eval()
    data, _ = next(iter(data_loader))
    input_data = data.view(-1, 784).to(device)
    
    with torch.no_grad():
        recon_batch, _, _ = model(input_data)
    
    plt.figure(figsize=(16,4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(input_data[i].cpu().view(28,28), cmap="gray")
        plt.axis("off")
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(recon_batch[i].cpu().view(28,28), cmap="gray")
        plt.axis("off")
    plt.suptitle("Original vs Reconstruction")
    if save_path:
        ensure_dir(save_path)
        plt.savefig(os.path.join(save_path, "reconstructions.png"))
    plt.show()

def show_generated_samples(model, latent_dim, device="cuda", n=16, save_path=None):
    model.eval()
    z = torch.randn(n, latent_dim).to(device)
    with torch.no_grad():
        samples = model.decode(z).cpu()
    
    plt.figure(figsize=(8,8))
    for i in range(n):
        ax = plt.subplot(4,4,i+1)
        plt.imshow(samples[i].view(28,28), cmap="gray")
        plt.axis("off")
    plt.suptitle("Random Generated Samples")
    if save_path:
        ensure_dir(save_path)
        plt.savefig(os.path.join(save_path, "generated_samples.png"))
    plt.show()

def latent_tsne_plot(model, data_loader, device="cuda", save_path=None):
    model.eval()
    zs, labels = [], []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(-1, 784).to(device)
            mu, _ = model.encode(data)
            zs.append(mu.cpu())
            labels.append(target)
    zs = torch.cat(zs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    zs_2d = TSNE(n_components=2).fit_transform(zs)
    
    plt.figure(figsize=(8,8))
    scatter = plt.scatter(zs_2d[:,0], zs_2d[:,1], c=labels, cmap="tab10", alpha=0.6, s=5)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE of Latent Space")
    if save_path:
        ensure_dir(save_path)
        plt.savefig(os.path.join(save_path, "latent_tsne.png"))
    plt.show()

def latent_traversal(model, latent_dim=20, device="cuda", steps=8, save_path=None):
    model.eval()
    z = torch.randn(1, latent_dim).to(device)
    fig, axes = plt.subplots(1, steps, figsize=(steps*2,2))
    with torch.no_grad():
        for i, alpha in enumerate(np.linspace(-3,3,steps)):
            z_mod = z.clone()
            z_mod[0,0] = alpha
            sample = model.decode(z_mod).cpu().view(28,28)
            axes[i].imshow(sample, cmap="gray")
            axes[i].axis("off")
    plt.suptitle("Latent Dimension Traversal")
    if save_path:
        ensure_dir(save_path)
        plt.savefig(os.path.join(save_path, "latent_traversal.png"))
    plt.show()

def reconstruction_error_heatmap(model, data_loader, device="cuda", save_path=None):
    model.eval()
    data, _ = next(iter(data_loader))
    input_data = data.view(-1, 784).to(device)
    with torch.no_grad():
        recon_batch, _, _ = model(input_data)
    diff = torch.abs(recon_batch[0].cpu() - input_data[0].cpu()).view(28,28)
    
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1)
    plt.imshow(input_data[0].cpu().view(28,28), cmap="gray")
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1,3,2)
    plt.imshow(recon_batch[0].cpu().view(28,28), cmap="gray")
    plt.title("Reconstruction")
    plt.axis("off")
    
    plt.subplot(1,3,3)
    plt.imshow(diff, cmap="hot")
    plt.title("Error Heatmap")
    plt.axis("off")
    
    if save_path:
        ensure_dir(save_path)
        plt.savefig(os.path.join(save_path, "reconstruction_error.png"))
    plt.show()

def kl_per_dimension_plot(mu, logvar, save_path=None):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_mean = kl.mean(0).cpu().numpy()
    
    plt.bar(range(len(kl_mean)), kl_mean)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Average KL")
    plt.title("KL Divergence per Latent Dimension")
    if save_path:
        ensure_dir(save_path)
        plt.savefig(os.path.join(save_path, "kl_per_dimension.png"))
    plt.show()