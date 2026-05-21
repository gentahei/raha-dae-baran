import torch
import torch.nn as nn

class DenoisingAutoEncoder(nn.Module):
  def __init__(self, input_dim: int, dropout: float = 0.2):
    super().__init__()

    self.latent_dim = 16

    self.encoder = nn.Sequential(
      nn.Linear(input_dim, 64),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, self.latent_dim)
    )

    self.decoder = nn.Sequential(
      nn.Linear(self.latent_dim, 32),
      nn.ReLU(),
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, input_dim)
    )

  def forward(self, x_noisy: torch.Tensor) -> torch.Tensor:
    z = self.encoder(x_noisy)
    x_hat = self.decoder(z)
    return x_hat
