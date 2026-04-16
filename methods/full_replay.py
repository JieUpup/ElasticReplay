import random
import torch

class ReplayBuffer:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.items = []

    def add_batch(self, batch):
        x_seq, adj_seq, y, task_id = batch
        bs = x_seq.size(0)
        for i in range(bs):
            item = (
                x_seq[i].detach().cpu(),
                adj_seq[i].detach().cpu(),
                y[i].detach().cpu(),
                task_id[i].detach().cpu(),
            )
            if len(self.items) < self.capacity:
                self.items.append(item)
            else:
                self.items.pop(0)
                self.items.append(item)

    def sample(self, n):
        n = min(n, len(self.items))
        return random.sample(self.items, n)

    def __len__(self):
        return len(self.items)

def collate_replay_samples(samples, device):
    x_seq = torch.stack([s[0] for s in samples], dim=0).to(device)
    adj_seq = torch.stack([s[1] for s in samples], dim=0).to(device)
    y = torch.stack([s[2] for s in samples], dim=0).to(device)
    task_id = torch.stack([s[3] for s in samples], dim=0).to(device)
    return x_seq, adj_seq, y, task_id
