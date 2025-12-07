import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from models_v3 import (
    GrbasDataset, GrbasDistDataset, PatientGrbasDistDataset,
    GrbasCNN, AutoGRBASModel
)

def compute_metrics(preds, labels, dim_names=["G", "R", "B", "A", "S"]):

    diff = preds - labels
    mae = np.mean(np.abs(diff), axis=0)
    mse = np.mean(diff ** 2, axis=0)
    rmse = np.sqrt(mse)

    mae_overall = np.mean(np.abs(diff))
    rmse_overall = np.sqrt(np.mean(diff ** 2))

    results = {}
    print(f"{'Item':<5} | {'MAE':<7} | {'RMSE':<7}")
    for i, name in enumerate(dim_names):
        print(f"{name:<5} | {mae[i]:.4f}  | {rmse[i]:.4f}")
        results[f"{name}_MAE"] = mae[i]
        results[f"{name}_RMSE"] = rmse[i]
    print(f"ALL   | {mae_overall:.4f}  | {rmse_overall:.4f}\n")

    return results

def compute_patient_level_metrics(preds, labels, patient_ids, dim_names=["G", "R", "B", "A", "S"]):
    unique_ids = np.unique(patient_ids)
    P = len(unique_ids)

    p_preds = np.zeros((P, 5), dtype=np.float32)
    p_labels = np.zeros((P, 5), dtype=np.float32)

    for i, pid in enumerate(unique_ids):
        mask = (patient_ids == pid)
        p_preds[i] = preds[mask].mean(axis=0)
        p_labels[i] = labels[mask].mean(axis=0)

    print("Patient-Level Aggregated Performance (/a, /i, /u Averaged)")
    return compute_metrics(p_preds, p_labels, dim_names)

def SLP_baseline(csv_path):
    print("[Human Baseline Analysis]")
    df = pd.read_csv(csv_path)
    items = ["G", "R", "B", "A", "S"]
    
    overall_diffs = []
    
    for slp in [1, 2, 3]:
        diffs = []
        for item in items:

            col_slp = f"SLP{slp}_{item}"
            col_mean = f"SLPall_{item}"
            
            if col_slp not in df.columns or col_mean not in df.columns:
                continue

            temp_df = df[[col_slp, col_mean]].dropna()
            
            x = temp_df[col_slp].values.astype(float)
            y = temp_df[col_mean].values.astype(float)
            
            diff = np.abs(x - y)
            diffs.append(diff.mean())
            overall_diffs.extend(diff)

        print(f"SLP{slp} vs Mean SLPall -> MAE: {np.mean(diffs):.4f}")

    print(f"Average Human Disagreement (Overall MAE): {np.mean(overall_diffs):.4f}")

def load_trained_model(args, device):
    print(f"Loading model from {args.checkpoint} (Mode: {args.mode})...")

    if args.mode == "cnn":
        model = GrbasCNN(num_vowels=3, use_vowel_embedding=True).to(device)
    elif args.mode == "crnn_patient":
        model = AutoGRBASModel(in_channels=3).to(device)
    elif args.mode == "crnn_single":
        model = AutoGRBASModel(in_channels=1).to(device)

    if "crnn" in args.mode:
        model.eval()
        n_mels = 80
        with torch.no_grad():
            dummy = torch.zeros(1, model.encoder.cnn[0].net[0].in_channels, n_mels, 100).to(device)
            _ = model.encoder(dummy)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "cnn":
        ds = GrbasDataset(csv_path=args.test_csv, normalize_labels=True)
    elif args.mode == "crnn_patient":
        ds = PatientGrbasDistDataset(csv_path=args.test_csv)
    elif args.mode == "crnn_single":
        ds = GrbasDistDataset(csv_path=args.test_csv)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_trained_model(args, device)

    all_preds = []
    all_labels = []
    all_patients = []

    print("Starting Inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            feat = batch["feat"].to(device)

            if args.mode == "cnn":
                label = batch["label"].to(device)
                vowel_id = batch["vowel_id"].to(device)
                pred = model(feat, vowel_id=vowel_id)

                if ds.normalize_labels:
                    pred = pred * ds.y_max
                    label = label * ds.y_max
            else:
                label = batch["target_mean"].to(device)
                _, pred = model(feat, return_scores=True)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_patients.extend(batch["patient_id"])

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    all_patients = np.array(all_patients)

    print(f"Mode: {args.mode} | Samples: {len(preds)}")
    compute_metrics(preds, labels)

    if args.mode != "crnn_patient":
        compute_patient_level_metrics(preds, labels, all_patients)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="cnn", choices=["cnn", "crnn_patient", "crnn_single"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="GRBAS_test.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--check_baseline", action="store_true")
    args = parser.parse_args()

    if args.check_baseline:
        SLP_baseline(args.test_csv)

    evaluate(args)