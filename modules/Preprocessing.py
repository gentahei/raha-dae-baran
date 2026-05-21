import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Preprocessing:
  
  def __init__(self, num_cols: list[str], cat_cols: list[str]):
    self.num_cols = num_cols
    self.cat_cols = cat_cols
    self.scaler = StandardScaler()
    self.one_hot_encoding = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    self.cat_dims: list[int] = []
    self.num_dim = 0
    self.total_dim = 0

  def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
    parts = []

    if self.num_cols:
      X_num = self.scaler.fit_transform(df[self.num_cols].values)
      self.num_dim = X_num.shape[1]
      parts.append(X_num)
    else:
      self.num_dim = 0

    if self.cat_cols:
      X_cat = self.one_hot_encoding.fit_transform(df[self.cat_cols].values)
      self.cat_dims = [len(cats) for cats in self.one_hot_encoding.categories_]
      parts.append(X_cat)
    else:
      self.cat_dims = []

    self.total_dim = sum(p.shape[1] for p in parts)
    return np.hstack(parts).astype(np.float32)
  
  def transform(self, df: pd.DataFrame) -> np.ndarray:
    parts = []

    if self.num_cols:
      X_num = self.scaler.transform(df[self.num_cols].values)
      parts.append(X_num)

    if self.cat_cols:
      X_cat = self.one_hot_encoding.transform(df[self.cat_cols].values)
      parts.append(X_cat)

    return np.hstack(parts).astype(np.float32)
  
  def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if self.num_cols:
      df[self.num_cols] = df[self.num_cols].fillna(df[self.num_cols].median())

    for col in self.cat_cols:
      mode = df[col].mode()
      df[col] = df[col].fillna(mode[0] if len(mode) > 0 else "unknown")

    return df