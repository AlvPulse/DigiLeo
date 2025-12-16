# dl_models.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

class AudioCNN(nn.Module):
    """
    Simple CNN for Audio Classification.
    Input shape: (Batch, 1, n_features, n_timesteps)
    """
    def __init__(self, n_features, n_timesteps, n_classes=2):
        super(AudioCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Calculate Flatten Size dynamically or hardcode if fixed
        # H_out = H_in / 8 (3 pooling layers)
        # W_out = W_in / 8
        h_out = n_features // 8
        w_out = n_timesteps // 8
        self.flatten_size = 64 * h_out * w_out

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, H, W) -> unsqueeze to (B, 1, H, W) if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class AudioDNN(nn.Module):
    """
    Simple MLP/DNN for Audio Classification.
    Input shape: (Batch, input_dim)
    """
    def __init__(self, input_dim, n_classes=2):
        super(AudioDNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-Learn Wrapper for PyTorch Models.
    Compatible with VotingClassifier.
    """
    def __init__(self, model_class, model_params, epochs=10, batch_size=32, lr=0.001, device='cpu'):
        self.model_class = model_class
        self.model_params = model_params
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # Determine Input Shape dynamically
        if self.model is None:
            if 'input_dim' not in self.model_params:
                # Infer from X
                if len(X.shape) == 2: # (N, Features)
                     self.model_params['input_dim'] = X.shape[1]
                elif len(X.shape) == 3: # (N, H, W)
                     self.model_params['n_features'] = X.shape[1]
                     self.model_params['n_timesteps'] = X.shape[2]

            self.model = self.model_class(**self.model_params).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Prepare Data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)

        # Drop last batch if it is 1 sample, to avoid BatchNorm errors
        drop_last = len(dataset) > self.batch_size
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        # Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {running_loss/len(loader):.4f}")

        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Batch inference to avoid OOM
        probas_list = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X_tensor[i:i+self.batch_size]
                outputs = self.model(batch)
                probas_list.append(torch.softmax(outputs, dim=1).cpu().numpy())

        return np.concatenate(probas_list, axis=0)
