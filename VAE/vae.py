import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    Implementation of Variational Autoencoder (VAE) in PyTorch.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()      
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # This will generate both mean and log-variance for the latent space
        )  
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        """
        The encoder function will learn to map the input data to a latent space and return its learned mean and log-variance.
        
        This will be q_phi (z | x) in the original VAE paper.

        Args:
            x : The original input data
        """
        
        mu, logvar = torch.chunk(self.encoder(x), 2, dim=-1)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Perform the reparameterization trick to sample a latent vector from the learned distribution of the latent space. 
        
        Args:
            mu: The mean of the latent space
            logvar: The log-variance of the latent space. We do this because it is numerically more stable to work with log-variance instead of variance.
        """
        eps = torch.randn_like(logvar)
        z = eps * torch.exp(0.5 * logvar) + mu
        
        return z
    
    def decode(self, z):
        """
        The decoder function will learn to map the latent vector back to the original input space.

        Args:
            z: The latent vector sampled from the latent space after reparameterization.
        """
        
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        reconstructed_x = self.decode(z)
        
        return reconstructed_x, mu, logvar