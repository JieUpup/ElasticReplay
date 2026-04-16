import os
import sys
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.dataset import StructuredToyBrainDataset
from data.stream_builder import build_task_stream_with_splits
from models.backbone import SimpleDynamicBrainNet
from utils.seed import set_seed
from engine.metrics import average_accuracy, final_average_accuracy, forgetting
from methods.full_replay import ReplayBuffer, collate_replay_samples
from methods.selective_replay import (
    SelectiveReplayBuffer,
    collate_selective_samples,
    select_topk_windows,
)


def collate_fn(batch):
    x_seq = torch.stack([b[0] for b in batch], dim=0)
    adj_seq = torch.stack([b[1] for b in batch], dim=0)
    y = torch.stack([b[2] for b in batch], dim=0)
    task_id = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return x_seq, adj_seq, y, task_id


def make_loader(ds, batch_size=32, shuffle=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for x_seq, adj_seq, y, _ in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        adj_seq = adj_seq.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x_seq, adj_seq)
        loss = criterion(logits, y)

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * y.size(0)

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return acc, avg_loss


def train_one_task(
    model,
    loader,
    optimizer,
    criterion,
    device,
    method="finetune",
    replay_buffer=None,
    replay_batch_size=32,
    topk_windows=2,
    epochs=20,
):
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0

        for x_seq, adj_seq, y, task_id in loader:
            x_seq = x_seq.to(device, non_blocking=True)
            adj_seq = adj_seq.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            task_id = task_id.to(device, non_blocking=True)

            logits = model(x_seq, adj_seq)
            loss = criterion(logits, y)

            if method == "full_replay" and replay_buffer is not None and len(replay_buffer) > 0:
                replay_samples = replay_buffer.sample(replay_batch_size)
                rx_seq, radj_seq, ry, _ = collate_replay_samples(replay_samples, device)
                rlogits = model(rx_seq, radj_seq)
                replay_loss = criterion(rlogits, ry)
                loss = loss + replay_loss

            if method == "selective_replay" and replay_buffer is not None and len(replay_buffer) > 0:
                replay_samples = replay_buffer.sample(replay_batch_size)
                rx_seq, radj_seq, ry, _ = collate_selective_samples(replay_samples, device)
                rlogits = model(rx_seq, radj_seq)
                replay_loss = criterion(rlogits, ry)
                loss = loss + replay_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total += bs

            if method == "full_replay" and replay_buffer is not None:
                replay_buffer.add_batch((x_seq, adj_seq, y, task_id))

            if method == "selective_replay" and replay_buffer is not None:
                selected_items = select_topk_windows(
                    model=model,
                    x_seq=x_seq.detach(),
                    adj_seq=adj_seq.detach(),
                    y=y.detach(),
                    task_id=task_id.detach(),
                    topk=topk_windows,
                    device=device,
                )
                replay_buffer.add_batch(selected_items)

        avg_loss = total_loss / total if total > 0 else 0.0
        history.append(avg_loss)
        print(f"  epoch={epoch:02d} loss={avg_loss:.4f}")

    return history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="finetune",
                        choices=["finetune", "full_replay", "selective_replay"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--samples_per_task", type=int, default=1000)
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--F", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--buffer_size", type=int, default=500)
    parser.add_argument("--replay_batch_size", type=int, default=32)
    parser.add_argument("--topk_windows", type=int, default=2)

    parser.add_argument("--noise_std", type=float, default=0.15)
    parser.add_argument("--task_shift", type=float, default=0.8)
    parser.add_argument("--class_signal", type=float, default=1.2)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print(f"method={args.method}")
    print(f"seed={args.seed}")

    os.makedirs("results", exist_ok=True)

    dataset = StructuredToyBrainDataset(
        num_tasks=args.num_tasks,
        samples_per_task=args.samples_per_task,
        T=args.T,
        N=args.N,
        F=args.F,
        num_classes=args.num_classes,
        noise_std=args.noise_std,
        task_shift=args.task_shift,
        class_signal=args.class_signal,
        seed=args.seed,
    )

   # streams = build_task_stream(dataset, num_tasks=args.num_tasks)
   # train_loaders = [make_loader(ds, batch_size=args.batch_size, shuffle=True) for ds in streams]
   # test_loaders = [make_loader(ds, batch_size=args.batch_size, shuffle=False) for ds in streams]
   train_streams, test_streams = build_task_stream_with_splits(
           dataset,
           num_tasks=args.num_tasks,
           train_ratio=0.8,
           )
   train_loaders = [make_loader(ds, batch_size=args.batch_size, shuffle=True) for ds in train_streams]
   test_loaders = [make_loader(ds, batch_size=args.batch_size, shuffle=False) for ds in test_streams]
   model = SimpleDynamicBrainNet(
        node_dim=args.F,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    replay_buffer = None
    if args.method == "full_replay":
        replay_buffer = ReplayBuffer(capacity=args.buffer_size)
    elif args.method == "selective_replay":
        replay_buffer = SelectiveReplayBuffer(capacity=args.buffer_size)

    num_tasks = len(streams)
    acc_matrix = [[0.0 for _ in range(num_tasks)] for _ in range(num_tasks)]
    loss_matrix = [[0.0 for _ in range(num_tasks)] for _ in range(num_tasks)]
    train_histories = {}

    for task_id in range(num_tasks):
        print(f"\n=== Train task {task_id} ===")
        history = train_one_task(
            model=model,
            loader=train_loaders[task_id],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            method=args.method,
            replay_buffer=replay_buffer,
            replay_batch_size=args.replay_batch_size,
            topk_windows=args.topk_windows,
            epochs=args.epochs,
        )
        train_histories[f"task_{task_id}"] = history

        print(f"--- Eval after task {task_id} ---")
        for eval_task in range(task_id + 1):
            acc, ev_loss = evaluate(model, test_loaders[eval_task], device)
            acc_matrix[task_id][eval_task] = acc
            loss_matrix[task_id][eval_task] = ev_loss
            print(f"eval_task={eval_task} acc={acc:.4f} loss={ev_loss:.4f}")

        avg_acc = average_accuracy(acc_matrix, task_id)
        print(f"avg_acc_after_task_{task_id}={avg_acc:.4f}")

        if replay_buffer is not None:
            print(f"buffer_size_now={len(replay_buffer)}")

    final_acc = final_average_accuracy(acc_matrix)
    final_forgetting = forgetting(acc_matrix)

    print("\n=== Final Summary ===")
    print("Accuracy Matrix:")
    for row in acc_matrix:
        print(["{:.4f}".format(x) for x in row])

    print(f"final_avg_acc={final_acc:.4f}")
    print(f"final_forgetting={final_forgetting:.4f}")

    out = {
        "method": args.method,
        "seed": args.seed,
        "config": vars(args),
        "acc_matrix": acc_matrix,
        "loss_matrix": loss_matrix,
        "final_avg_acc": final_acc,
        "final_forgetting": final_forgetting,
        "train_histories": train_histories,
    }

    out_name = (
        f"{args.method}"
        f"_seed{args.seed}"
        f"_spt{args.samples_per_task}"
        f"_ep{args.epochs}.json"
    )
    out_path = os.path.join("results", out_name)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"saved_to={out_path}")


if __name__ == "__main__":
    main()
