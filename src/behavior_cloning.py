import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BCActor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, output_dim), nn.Tanh()  # Assuming output normalized to [-1,1]
        )

    def forward(self, x):
        return self.net(x)

def train_bc(expert_dataset, epochs=100, batch_size=64, lr=1e-3):
    # Convert dataset list into tensors
    obs = torch.tensor(np.array([item[0] for item in expert_dataset]), dtype=torch.float32)
    acts = torch.tensor(np.array([item[1] for item in expert_dataset]), dtype=torch.float32)

    dataset = TensorDataset(obs, acts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = obs.shape[1]
    output_dim = acts.shape[1]
    model = BCActor(input_dim=input_dim, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_acts in dataloader:
            pred = model(batch_obs)
            loss = criterion(pred, batch_acts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_obs.size(0)
        avg_loss = total_loss / len(dataset)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return model

