import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models_v3 import (
    GrbasDataset, GrbasDistDataset, PatientGrbasDistDataset,
    GrbasCNN, AutoGRBASModel, grbas_distribution_loss
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="crnn_patient", choices=["cnn", "crnn_patient", "crnn_single"])
    parser.add_argument("--train_csv", type=str, default="GRBAS_train.csv")
    parser.add_argument("--val_csv", type=str, default="GRBAS_val.csv")
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_reg", type=float, default=0.3)
    
    return parser.parse_args()

def grbas_hybrid_loss(logits, target_dist, target_mean, score_values, lambda_reg=0.3):

    ce_loss = grbas_distribution_loss(logits, target_dist)
    probs = F.softmax(logits, dim=-1)

    expected_scores = torch.sum(probs * score_values.view(1, 1, -1), dim=-1)
    reg_loss = F.l1_loss(expected_scores, target_mean)

    total_loss = ce_loss + lambda_reg * reg_loss
    return total_loss, ce_loss, reg_loss

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode}")

    if args.mode == "cnn":
        train_ds = GrbasDataset(csv_path=args.train_csv, normalize_labels=True)
        val_ds   = GrbasDataset(csv_path=args.val_csv, normalize_labels=True)
        model = GrbasCNN(num_vowels=3, use_vowel_embedding=True).to(device)
        criterion = nn.MSELoss() 
        
    elif args.mode == "crnn_patient":
        train_ds = PatientGrbasDistDataset(csv_path=args.train_csv)
        val_ds   = PatientGrbasDistDataset(csv_path=args.val_csv)
        model = AutoGRBASModel(in_channels=3).to(device)
        
    elif args.mode == "crnn_single":
        train_ds = GrbasDistDataset(csv_path=args.train_csv)
        val_ds   = GrbasDistDataset(csv_path=args.val_csv)
        model = AutoGRBASModel(in_channels=1).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    score_values = torch.arange(1, 6, dtype=torch.float32).to(device)

    best_val_mae = float('inf')
    early_stop_counter = 0
    early_stop = 50

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            feat = batch["feat"].to(device)
            optimizer.zero_grad()
            loss = 0.0

            if args.mode == "cnn":
                label = batch["label"].to(device)
                vowel_id = batch["vowel_id"].to(device)
                pred = model(feat, vowel_id=vowel_id)
                loss = criterion(pred, label)

            else:
                target_dist = batch["target_dist"].to(device)
                target_mean = batch["target_mean"].to(device)

                logits = model(feat)
                loss, ce, reg = grbas_hybrid_loss(logits, target_dist, target_mean, score_values, args.lambda_reg)

            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * feat.size(0)
        train_loss_avg = train_loss_accum / len(train_ds)

        model.eval()
        val_loss_accum = 0.0
        val_mae_accum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                feat = batch["feat"].to(device)
                bs = feat.size(0)

                if args.mode == "cnn":
                    label = batch["label"].to(device)
                    vowel_id = batch["vowel_id"].to(device)

                    pred = model(feat, vowel_id=vowel_id)
                    loss = criterion(pred, label)

                    if train_ds.normalize_labels:
                        pred_score = pred * 5.0
                        label_score = label * 5.0
                    else:
                        pred_score = pred
                        label_score = label

                    val_mae_accum += torch.abs(pred_score - label_score).mean().item() * bs

                else:
                    target_dist = batch["target_dist"].to(device)
                    target_mean = batch["target_mean"].to(device)

                    logits = model(feat)
                    loss, _, _ = grbas_hybrid_loss(logits, target_dist, target_mean, score_values, args.lambda_reg)

                    probs = F.softmax(logits, dim=-1)
                    expected_scores = torch.sum(probs * score_values.view(1, 1, -1), dim=-1)
                    val_mae_accum += torch.abs(expected_scores - target_mean).mean().item() * bs

                val_loss_accum += loss.item() * bs

        val_loss_avg = val_loss_accum / len(val_ds)
        val_mae_avg  = val_mae_accum / len(val_ds)

        print(f"[Epoch {epoch:03d}] Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Val MAE: {val_mae_avg:.4f}")

        if val_mae_avg < best_val_mae:
            best_val_mae = val_mae_avg
            early_stop_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae_avg,
                "config": args
            }, args.save_path)
            print(f"  >>> Best model saved! (MAE: {val_mae_avg:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()