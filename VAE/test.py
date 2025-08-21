import torch
from vae import VAE
from utils import get_dataloaders, show_reconstructions, show_generated_samples, latent_tsne_plot, latent_traversal, reconstruction_error_heatmap, kl_per_dimension_plot
import argparse

def test_vae(dataset_name="MNIST", input_dim=784, hidden_dim=400, latent_dim=20,
             batch_size=64, device="cuda", checkpoint=None, visualize=True):
    
    _, test_loader = get_dataloaders(dataset_name, batch_size)
    
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint from {checkpoint}")
    
    model.eval()
    
    if visualize:
        # Reconstructions
        show_reconstructions(model, test_loader, device)
        
        # Random generations
        show_generated_samples(model, latent_dim, device)
        
        # Latent t-SNE
        latent_tsne_plot(model, test_loader, device)
        
        # Latent traversal
        latent_traversal(model, latent_dim, device)
        
        # Reconstruction error heatmap
        reconstruction_error_heatmap(model, test_loader, device)
        
        # KL per dimension (requires passing a batch)
        data_iter = iter(test_loader)
        data, _ = next(data_iter)
        data = data.view(-1, input_dim).to(device)
        with torch.no_grad():
            mu, logvar = model.encode(data)
        kl_per_dimension_plot(mu, logvar)

def main():
    parser = argparse.ArgumentParser(description="Test a trained VAE")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST","FashionMNIST"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=400)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    _, test_loader = get_dataloaders(args.dataset, args.batch_size)
    
    model = VAE(input_dim=784, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(args.device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    model.eval()
    
    save_dir = "./vae_results"

    show_reconstructions(model, test_loader, device="cuda", save_path=save_dir)
    show_generated_samples(model, args.latent_dim, device="cuda", save_path=save_dir)
    latent_tsne_plot(model, test_loader, device="cuda", save_path=save_dir)
    latent_traversal(model, args.latent_dim, device="cuda", save_path=save_dir)
    reconstruction_error_heatmap(model, test_loader, device="cuda", save_path=save_dir)
    
    # KL per dimension
    data, _ = next(iter(test_loader))
    data = data.view(-1, 784).to(args.device)
    with torch.no_grad():
        mu, logvar = model.encode(data)
    kl_per_dimension_plot(mu, logvar, save_path=save_dir)


if __name__ == "__main__":
    main()