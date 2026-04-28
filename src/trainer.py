import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler, autocast

from data_loader import generate_chess_batches
from network import ChessNet

# -----------------------------------------------------------------------------
# Core Settings
# -----------------------------------------------------------------------------
PGN_FILE = "../Data/training_data_eval_1800_plus.pgn"
TACTICS_FILE = "../Data/tactics.pgn"
CHECKPOINT_DIR = "../checkpoints_V5_Large"

EPOCHS = 15
BATCH_SIZE = 1024
LR = 0.000710
WEIGHT_DECAY = 0.000001
VALUE_LOSS_WEIGHT = 0.660273
LOG_EVERY = 50

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize the scaled-up V5 model
    model = ChessNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    model.train()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Starting Epoch {epoch}/{EPOCHS} ---")
        epoch_start = time.perf_counter()
        
        total_loss = total_policy = total_value = 0.0
        samples = 0

        # Injecting the tactics file for blended batch generation
        batch_generator = generate_chess_batches(
            PGN_FILE, 
            batch_size=BATCH_SIZE,
            tactics_path=TACTICS_FILE
        )

        for i, (x_batch, p_batch, v_batch) in enumerate(batch_generator, 1):
            x = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            p_targets = torch.as_tensor(p_batch, dtype=torch.long, device=device)
            v_targets = torch.as_tensor(v_batch, dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                policy_logits, pred_value = model(x)
                
                p_loss = policy_criterion(policy_logits, p_targets)
                v_loss = value_criterion(pred_value, v_targets)
                loss = p_loss + (VALUE_LOSS_WEIGHT * v_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_policy += p_loss.item()
            total_value += v_loss.item()
            samples += x.size(0)

            if i % LOG_EVERY == 0:
                print(
                    f"Batch {i} | "
                    f"Loss: {total_loss/i:.4f} | "
                    f"Policy: {total_policy/i:.4f} | "
                    f"Value: {total_value/i:.4f}"
                )

        # Save inference weights at the end of each epoch
        save_path = os.path.join(CHECKPOINT_DIR, f"chess_model_epoch_{epoch}_V5.pth")
        torch.save(model.state_dict(), save_path)

        elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch} Complete | "
            f"Samples/sec: {samples / elapsed:.1f} | "
            f"Saved: {save_path}"
        )

if __name__ == "__main__":
    train()