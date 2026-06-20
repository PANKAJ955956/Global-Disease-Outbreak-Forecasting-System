import os
import sqlite3
import logging
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DB_PATH, MODELS_DIR, SEQUENCE_LENGTH, RANDOM_STATE

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_lstm")

# Check device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define the 3-layer Stacked LSTM Model in PyTorch
class StackedLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=3, output_dim=1, dropout=0.2):
        super(StackedLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 3-layer stacked LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dense layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def prepare_sequences(df):
    """
    Groups data by country and disease, fits a MinMaxScaler for case rates,
    and constructs 12-week input sequences for LSTM training.
    """
    logger.info("Preparing 12-week time-series sequences grouped by country and disease...")
    
    # We will fit a single global MinMaxScaler for case_rate
    scaler = MinMaxScaler()
    df["case_rate_scaled"] = scaler.fit_transform(df[["case_rate"]])
    
    # Save the LSTM case rate scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
    
    X_list, y_list = [], []
    
    grouped = df.groupby(["country", "disease"])
    for name, group in grouped:
        group_sorted = group.sort_values("date")
        rates = group_sorted["case_rate_scaled"].values
        
        if len(rates) <= SEQUENCE_LENGTH:
            continue
            
        for i in range(len(rates) - SEQUENCE_LENGTH):
            X_list.append(rates[i : i + SEQUENCE_LENGTH])
            y_list.append(rates[i + SEQUENCE_LENGTH])
            
    X = np.array(X_list).reshape(-1, SEQUENCE_LENGTH, 1)
    y = np.array(y_list).reshape(-1, 1)
    
    return X, y, scaler

def main():
    logger.info("Initializing LSTM Deep Learning Pipeline")
    
    # 1. Load data from SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM features", conn)
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        logger.error(f"Error loading features from SQLite: {e}")
        raise e
    finally:
        conn.close()
        
    # 2. Sequence preparation
    X, y, scaler = prepare_sequences(df)
    logger.info(f"Prepared sequence data. Input shape: {X.shape}, Target shape: {y.shape}")
    
    # 3. Train-Test Split (80% train, 20% validation)
    # We split chronologically or randomly. For training robustness, we'll do random split of sequences.
    np.random.seed(RANDOM_STATE)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx = int(len(X) * 0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 4. Instantiate Model
    model = StackedLSTM(input_dim=1, hidden_dim=64, num_layers=3, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    # 5. Training Loop
    epochs = 20  # Fast execution in workspace sandbox, but sufficient to converge on simulated data
    best_loss = float("inf")
    patience = 5
    patience_counter = 0
    
    logger.info("Training PyTorch Stacked LSTM model...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation epoch evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_loss = criterion(val_outputs, y_val_t.to(device)).item()
            
        scheduler.step(val_loss)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early Stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm_model.pt"))
            logger.info("Saved best model weights.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break
                
    # 6. Evaluation metrics on Validation set
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm_model.pt")))
    model.eval()
    with torch.no_grad():
        val_preds_scaled = model(X_val_t.to(device)).cpu().numpy()
        
    # Inverse transform predictions and targets to original scale (case rate per 100k)
    y_val_actual = scaler.inverse_transform(y_val)
    y_val_pred = scaler.inverse_transform(val_preds_scaled)
    
    # Clip predictions to prevent negative forecast rates
    y_val_pred = np.clip(y_val_pred, 0.0, None)
    
    mae = mean_absolute_error(y_val_actual, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
    mape = np.mean(np.abs((y_val_actual - y_val_pred) / (y_val_actual + 1e-5))) * 100.0
    
    logger.info(f"Validation MAE: {mae:.4f}")
    logger.info(f"Validation RMSE: {rmse:.4f}")
    logger.info(f"Validation MAPE: {mape:.2f}%")
    
    # Save validation metrics
    metrics = {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}
    joblib.dump(metrics, os.path.join(MODELS_DIR, "lstm_metrics.pkl"))
    logger.info("LSTM model training complete and metrics logged.")

if __name__ == "__main__":
    main()
