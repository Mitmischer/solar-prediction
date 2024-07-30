import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Random Forest Feature Selection
class TimeSeriesRandomForest(nn.Module):
    def __init__(self, n_features, n_estimators=100, max_depth=10):
        super().__init__()
        self.n_estimators = n_estimators
        self.trees = nn.ModuleList([nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ) for _ in range(n_estimators)])
        self.feature_importances_ = None  # initialize during forward pass

    def forward(self, x):
        if self.feature_importances_ is None:
            self.feature_importances_ = torch.zeros(x.shape[1], device=x.device)
        return torch.mean(torch.stack([tree(x) for tree in self.trees]), dim=0)

    def update_feature_importances(self):
        for tree in self.trees:
            self.feature_importances_ += torch.abs(tree[0].weight.data.sum(dim=0))
        self.feature_importances_ /= self.n_estimators


def time_series_feature_selection_gpu(train_data, train_target, n_features=50, n_estimators=100, max_depth=10,
                                      epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = torch.tensor(train_data.values, dtype=torch.float32).to(device)
    y = torch.tensor(train_target.values, dtype=torch.float32).unsqueeze(1).to(device)

    model = TimeSeriesRandomForest(X.shape[1], n_estimators, max_depth).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    tscv = TimeSeriesSplit(n_splits=5)

    for epoch in range(epochs):
        epoch_loss = 0
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)
                epoch_loss += val_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {epoch_loss / 5:.4f}")

    model.update_feature_importances()
    importances = model.feature_importances_.cpu().numpy()

    indices = np.argsort(importances)[::-1]
    selected_indices = indices[:n_features]
    selected_features = train_data.columns[selected_indices].tolist()

    return train_data[selected_features], selected_features
