import random
import torch
import torch.nn.functional as F

class SelectiveReplayBuffer:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.items = []

    def add_item(self, item):
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items.pop(0)
            self.items.append(item)

    def add_batch(self, selected_items):
        for item in selected_items:
            self.add_item(item)

    def sample(self, n):
        n = min(n, len(self.items))
        return random.sample(self.items, n)

    def __len__(self):
        return len(self.items)

def collate_selective_samples(samples, device):
    x_seq = torch.stack([s[0] for s in samples], dim=0).to(device)
    adj_seq = torch.stack([s[1] for s in samples], dim=0).to(device)
    y = torch.stack([s[2] for s in samples], dim=0).to(device)
    task_id = torch.stack([s[3] for s in samples], dim=0).to(device)
    return x_seq, adj_seq, y, task_id

@torch.no_grad()
def select_topk_windows(model, x_seq, adj_seq, y, task_id, topk=2, device="cuda"):
    model.eval()
    bs, T = x_seq.size(0), x_seq.size(1)
    selected = []

    for i in range(bs):
        scores = []
        for t in range(T):
            x_win = x_seq[i:i+1, t:t+1].to(device)
            a_win = adj_seq[i:i+1, t:t+1].to(device)
            logits = model(x_win, a_win)
            prob = F.softmax(logits, dim=-1)
            entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1)
            scores.append((float(entropy.item()), t))

        scores.sort(reverse=True)
        top_indices = [t for _, t in scores[:topk]]

        for t in top_indices:
            item = (
                x_seq[i, t:t+1].detach().cpu(),
                adj_seq[i, t:t+1].detach().cpu(),
                y[i].detach().cpu(),
                task_id[i].detach().cpu(),
            )
            selected.append(item)

    return selected
