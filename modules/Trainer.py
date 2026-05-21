import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader
from modules.Network import DenoisingAutoEncoder

class DAETrainer:
  def __init__(
    self,
    prep,
    lr: float = 1e-2,
    patience: int = 10,
    min_delta: float = 1e-4,
    max_epoch: int = 100,
    device: torch.device = torch.device("cpu"),
  ):
    self.prep = prep
    self.patience = patience
    self.min_delta = min_delta
    self.max_epoch = max_epoch
    self.device = device

    self.model = DenoisingAutoEncoder(prep.total_dim).to(device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
    self.noise_fn = self._add_swap_noise

    self.history = []
    self.best_loss = float("inf")
    self.best_epoch = 1

  def _add_swap_noise(self, x: torch.Tensor, proba: float = 0.15):
    should_swap = torch.bernoulli(torch.ones_like(x) * proba)
    shuffled = x[torch.randperm(x.shape[0])]
    return torch.where(should_swap.bool(), shuffled, x)

  def _compute_loss(self, x_hat, x_clean, cat_weight=1.0):
    mse_loss = nn.MSELoss()(x_hat[:, :self.prep.num_dim], x_clean[:, :self.prep.num_dim])

    nll_loss = torch.tensor(0.0, device=x_hat.device)
    cursor = self.prep.num_dim
    for dim in self.prep.cat_dims:
      logits = x_hat[:, cursor : cursor + dim]
      target = x_clean[:, cursor : cursor + dim].argmax(dim=1)
      probs = torch.softmax(logits, dim=1)
      nll = -torch.log(probs[torch.arange(len(target), device=x_hat.device), target] + 1e-8)
      nll_loss += nll.mean() / len(self.prep.cat_dims)
      cursor += dim

    return mse_loss + cat_weight * nll_loss, mse_loss, nll_loss

  def _train_epoch(self, loader: DataLoader):
    self.model.train()
    total = 0.0
    for (x_clean,) in loader:
      x_clean = x_clean.to(self.device)
      x_noisy = self.noise_fn(x_clean)

      self.optimizer.zero_grad()
      x_hat = self.model(x_noisy)
      loss, mse, nll = self._compute_loss(x_hat, x_clean)
      loss.backward()
      self.optimizer.step()

      total += loss.item() * len(x_clean)

    return total / len(loader.dataset), mse, nll

  def fit(self, loader: DataLoader) -> pd.DataFrame:
    patience_count = 0
    best_state = None
    self.history = []

    for epoch in range(1, self.max_epoch + 1):
      loss, mse, nll = self._train_epoch(loader)
      self.scheduler.step()

      self.history.append({
        "epoch": epoch,
        "loss": loss,
        "mse": mse.item(),
        "nll": nll.item(),
        "lr": self.optimizer.param_groups[0]["lr"],
      })

      if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{self.max_epoch} | Loss: {loss:.4f} | MSE: {mse:.4f} | NLL: {nll:.4f}")

      if loss < self.best_loss - self.min_delta:
        self.best_loss = loss
        self.best_epoch = epoch
        patience_count = 0
        best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
      else:
        patience_count += 1
        if patience_count >= self.patience:
          print(f"Early stopping epoch {epoch} | Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")
          break

    if best_state is not None:
      self.model.load_state_dict(best_state)
      print(f"Model restored to epoch {self.best_epoch}")

    return pd.DataFrame(self.history)