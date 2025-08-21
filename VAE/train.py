import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from vae import VAE
from utils import get_dataloaders, loss_function
import argparse

def train_vae(dataset_name="MNIST", input_dim=784, hidden_dim=400, latent_dim=20, 
              batch_size=128, lr=1e-3, epochs=10, device="cuda"):
    
    # Load data
    train_loader, test_loader = get_dataloaders(dataset_name, batch_size)
    
    # Model + optimizer
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track losses
    epoch_losses, recon_losses, kl_losses = [], [], []
    
    model.train()
    for epoch in range(epochs):
        train_loss, recon_loss_epoch, kl_loss_epoch = 0, 0, 0
        
        # tqdm progress bar per epoch
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (data, _) in enumerate(loop):
            data = data.view(-1, input_dim).to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            BCE, KLD = loss_function(recon_batch, data, mu, logvar, return_parts=True)
            loss = BCE + KLD
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_epoch += BCE.item()
            kl_loss_epoch += KLD.item()
            
            # Update tqdm postfix
            loop.set_postfix(TotalLoss=loss.item()/len(data), Recon=BCE.item()/len(data), KL=KLD.item()/len(data))
        
        avg_loss = train_loss / len(train_loader.dataset) #type:ignore
        avg_recon = recon_loss_epoch / len(train_loader.dataset) #type:ignore
        avg_kl = kl_loss_epoch / len(train_loader.dataset)#type:ignore
        
        epoch_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        
        print(f"Epoch {epoch+1}/{epochs} → Total: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
    
    # Save model
    torch.save(model.state_dict(), f"vae_{dataset_name.lower()}.pth")
    print("Model saved.")
    
    # Plot loss curves
    plt.figure(figsize=(10,5))
    plt.plot(epoch_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kl_losses, label="KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.show()
    
    return model, test_loader

def main():
    parser = argparse.ArgumentParser(description="Train a VAE on MNIST or FashionMNIST")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST","FashionMNIST"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=400)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Train the VAE
    model, test_loader = train_vae(
        dataset_name=args.dataset,
        input_dim=784,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device
    )

if __name__ == "__main__":
    main()