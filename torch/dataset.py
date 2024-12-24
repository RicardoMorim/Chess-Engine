import torch


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float16)  # Use float16 for memory efficiency
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]