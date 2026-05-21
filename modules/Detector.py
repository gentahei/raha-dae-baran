import torch
import numpy as np
import pandas as pd
from scipy.stats import norm

class Detector:
  def __init__(self, trainer, prep):
    self.trainer = trainer
    self.prep = prep
    self.params_per_col = {}
    self.df_cell_error = None
    self.df_p_values = None
    self.df_cell_flag = None

  def _error_cols(self):
    return self.prep.num_cols + self.prep.cat_cols

  def _compute_error(self, X_tensor):
    self.trainer.model.eval()
    with torch.no_grad():
      x_clean = X_tensor.to(self.trainer.device)
      x_hat = self.trainer.model(x_clean)

      # MSE per sel
      mse_per_cell = (x_hat[:, :self.prep.num_dim] - x_clean[:, :self.prep.num_dim]) ** 2

      # NLL per sel
      nll_per_cell = []
      cursor = self.prep.num_dim
      for dim in self.prep.cat_dims:
        logits = x_hat[:, cursor : cursor + dim]
        target = x_clean[:, cursor : cursor + dim].argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
        nll = -torch.log(probs[torch.arange(len(target), device=x_hat.device), target] + 1e-8)
        nll_per_cell.append(nll.unsqueeze(1))
        cursor += dim

      if nll_per_cell:
        nll_per_cell = torch.cat(nll_per_cell, dim=1)
        error = torch.cat([mse_per_cell, nll_per_cell], dim=1)
      else:
        error = mse_per_cell

    return error.cpu().numpy()

  def fit_clean(self, X_tensor):
    error_clean = self._compute_error(X_tensor)
    error_cols = self._error_cols()

    for i, col in enumerate(error_cols):
      mu, std = norm.fit(error_clean[:, i])
      self.params_per_col[col] = (mu, std)

    return self

  def detect(self, df_dirty, df_clean, p_threshold=0.05):
    error_cols = self._error_cols()

    # preprocessing dirty
    df_dirty_filled = df_dirty.copy()

    for col in self.prep.num_cols:
      df_dirty_filled[col] = pd.to_numeric(df_dirty_filled[col], errors="coerce")

    df_dirty_filled[self.prep.num_cols] = df_dirty_filled[self.prep.num_cols].fillna(
      df_clean[self.prep.num_cols].median()
    )
    
    for col in self.prep.cat_cols:
      mode = df_clean[col].mode()
      df_dirty_filled[col] = df_dirty_filled[col].fillna(
        mode[0] if len(mode) > 0 else "unknown"
      )

    # transform
    X_dirty  = self.prep.transform(df_dirty_filled)
    X_dirty_tensor = torch.tensor(X_dirty)

    # error per cell
    error_per_cell = self._compute_error(X_dirty_tensor)
    self.df_cell_error = pd.DataFrame(error_per_cell, columns=error_cols)

    # reconstruction probability
    p_values = np.zeros_like(error_per_cell)
    for i, col in enumerate(error_cols):
      mu, std  = self.params_per_col[col]
      p_values[:, i]  = 1 - norm.cdf(error_per_cell[:, i], mu, std)

    self.df_p_values  = pd.DataFrame(p_values, columns=error_cols)
    self.df_cell_flag = self.df_p_values < p_threshold

    return self

  def get_detected_cells(self, df_dirty):
    col_to_idx     = {col: idx for idx, col in enumerate(df_dirty.columns)}
    detected_cells = {}

    for col in self.df_cell_flag.columns:
      if col not in col_to_idx:
        continue
      col_idx      = col_to_idx[col]
      flagged_rows = self.df_cell_flag.index[self.df_cell_flag[col]].tolist()
      for row_idx in flagged_rows:
        detected_cells[(row_idx, col_idx)] = "JUST A DUMMY VALUE" # Baran Format

    return detected_cells

  def summary(self):
    sum_cell_flag    = self.df_cell_flag.sum().sort_values(ascending=False)
    df_sum_cell_flag = pd.DataFrame({
      "col_name"    : sum_cell_flag.index,
      "total"       : sum_cell_flag.values,
      "p_value_mean": [self.df_p_values[col].mean() for col in sum_cell_flag.index],
    })

    anomaly_ratio    = f"{self.df_cell_flag.sum().sum() / self.df_cell_flag.size * 100:.2f}%"
    df_anomaly_result = pd.DataFrame({
      "description": ["anomaly", "total_cell", "ratio"],
      "values"     : [self.df_cell_flag.sum().sum(), self.df_cell_flag.size, anomaly_ratio]
    })

    return df_sum_cell_flag, df_anomaly_result