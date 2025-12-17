# dl_models.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import inspect
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
    Input shape:
        - (Batch, input_dim) if flattened
        - (Batch, n_features, n_timesteps) if 2D (will be flattened)
    """
    def __init__(self, n_features=None, n_timesteps=None, input_dim=None, n_classes=2):
        super(AudioDNN, self).__init__()

        # Determine actual input dimension
        if input_dim is not None:
            self.flat_dim = input_dim
        elif n_features is not None and n_timesteps is not None:
            self.flat_dim = n_features * n_timesteps
        else:
            raise ValueError("AudioDNN requires either 'input_dim' or both 'n_features' and 'n_timesteps'.")

        self.net = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
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
        # Flatten if input is 2D (or 3D tensor: B, F, T)
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.net(x)

class AudioLSTM(nn.Module):
    """
    LSTM for Audio Classification on FBE sequences.
    Input: (Batch, Features, Time) from loader -> Transformed to (Batch, Time, Features) for LSTM
    """
    def __init__(self, n_features, n_timesteps, n_classes=2, hidden_size=64, num_layers=2):
        super(AudioLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM
        # batch_first=True -> Input expected: (Batch, SeqLen, Features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # Fully Connected Classifier
        self.fc = nn.Linear(hidden_size, n_classes)

        # Note: User asked for Sigmoid output.
        # Since we use CrossEntropyLoss (standard for this pipeline), we output logits.
        # This is mathematically equivalent to Sigmoid for binary tasks if we take Softmax
        # over 2 classes, or we can use BCEWithLogitsLoss if we had 1 output.
        # To maintain compatibility with the pipeline's predict_proba, we keep n_classes output.

    def forward(self, x):
        # x shape: (Batch, n_features, n_timesteps) typically from our loader
        # LSTM expects: (Batch, n_timesteps, n_features)

        # Check dim and permute if needed
        if x.dim() == 3:
             # Assume (B, F, T). Check if F matches n_features
             # If n_features is 20 (n_mels), and shape is (B, 20, 32), then F is dim 1.
             # We want (B, T, F) -> permute(0, 2, 1)
             x = x.permute(0, 2, 1)

        # LSTM Output
        # out: (Batch, SeqLen, Hidden)
        # hn: (NumLayers, Batch, Hidden)
        out, (hn, cn) = self.lstm(x)

        # We can take the last time step output or the last hidden state
        # out[:, -1, :] is the output of the last time step
        last_out = out[:, -1, :]

        logits = self.fc(last_out)
        return logits

class SaraCNN(nn.Module):
    """
    Port of Sara Al-Emadi's CNN.
    Adapted for 2D Audio Inputs (N_MFCC x Time).
    Original Architecture:
      - Conv1D(32) -> MaxPool
      - Conv1D(64) -> MaxPool
      - Conv1D(128) -> MaxPool
      - Conv1D(128) -> MaxPool
      - Dense(256) -> Sigmoid -> Dense(2)
    """
    def __init__(self, n_features, n_timesteps, n_classes=2):
        super(SaraCNN, self).__init__()

        # We treat n_features (MFCC bins) as the "Input Channels" for Conv1D?
        # OR we treat the whole thing as a sequence of length T with F channels?
        # Usually for Audio:
        # Input: (Batch, Channels=n_mfcc, Time)
        # Conv1D scans across Time.

        # Sara's code: Reshape((x.shape[1], 1)) -> Input is 1 Channel, Sequence Length = x.shape[1]
        # This implies she was feeding a FLATTENED vector or raw audio.
        # But if we feed MFCCs (F x T), it's best to treat F as Channels.

        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),

            # Block 4
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),

            nn.Dropout(0.25)
        )

        # Calculate Flatten Size
        # We iterate to find output size
        dummy_input = torch.zeros(1, n_features, n_timesteps)
        with torch.no_grad():
            dummy_out = self.conv_blocks(dummy_input)
        self.flatten_size = dummy_out.numel()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(), # Sara used 'relu' (inner_activation_fun)
            nn.Linear(256, n_classes)
            # Note: Sara used sigmoid for multi-label or binary.
            # We use CrossEntropyLoss which expects logits (no sigmoid/softmax here).
        )

    def forward(self, x):
        # Expects (Batch, Features, Time)
        # If input is (Batch, 1, Features, Time), squeeze the channel dim
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        return self.fc(self.conv_blocks(x))

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
            # We work on a copy to avoid mutating the original dict which might be used elsewhere
            # or in subsequent fits (though fit clears model, better safe)
            init_params = self.model_params.copy()

            if 'input_dim' not in init_params:
                # Infer from X
                if len(X.shape) == 2: # (N, Features)
                     init_params['input_dim'] = X.shape[1]
                elif len(X.shape) == 3: # (N, H, W)
                     init_params['n_features'] = X.shape[1]
                     init_params['n_timesteps'] = X.shape[2]

            # Filter params to only those accepted by model_class.__init__
            sig = inspect.signature(self.model_class.__init__)
            valid_keys = [
                p.name for p in sig.parameters.values()
                if p.name != 'self' and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            ]

            # Check for **kwargs support (VAR_KEYWORD)
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

            if not has_var_keyword:
                filtered_params = {k: v for k, v in init_params.items() if k in valid_keys}
            else:
                filtered_params = init_params

            # Final check to ensure required params are present
            # If still missing (e.g. because X was 2D but model needs 3D dims),
            # we try to inject defaults or raise a clearer error.
            if 'n_features' in valid_keys and 'n_features' not in filtered_params:
                # Check if 'n_features' is mandatory (no default value)
                param = sig.parameters['n_features']
                if param.default == inspect.Parameter.empty:
                    raise ValueError(
                        f"Model {self.model_class.__name__} requires 'n_features', but it was not provided "
                        f"and could not be inferred from input shape {X.shape}. "
                        "Please provide 'n_features' (and 'n_timesteps') in 'model_params' or ensure input is 3D."
                    )

            self.model = self.model_class(**filtered_params).to(self.device)
        
        pos_weight = torch.tensor([2.0]).to(self.device) 

        # Use BCEWithLogitsLoss for binary classification (Standard for 2-class DL)
        # It's more stable than CrossEntropy for binary problems
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        #criterion = nn.CrossEntropyLoss()
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
        stopper = EarlyStopping(patience=3) # Stop after 3 bad epochs
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

            avg_val_loss = running_loss / len(loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {running_loss/len(loader):.4f}")
            # Check Early Stopping
            stopper(avg_val_loss)
            if stopper.early_stop:
                print("🛑 Early stopping triggered!")
                break

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
    
    # Add this class at the top of src/dl_models.py
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
