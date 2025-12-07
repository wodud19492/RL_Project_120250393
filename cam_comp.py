import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from models_v3 import AutoGRBASModel, PatientGrbasDistDataset
from gradcam import GrbasGradCAM
from rlcam import GrbasRLCAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_crnn_5587.pt"
CSV_PATH = "GRBAS_test.csv"
SAVE_DIR = "results_comparison5"
ITEMS = ["G", "R", "B", "A", "S"]

STEPS = 150
LAMBDA_REG = 0.1
LR = 0.1

def load_model():
    print(f"Loading Model from {CKPT_PATH}...")
    model = AutoGRBASModel(in_channels=3).to(DEVICE)

    dummy = torch.zeros(1, 3, 80, 100).to(DEVICE)
    model.eval()
    model.encoder(dummy)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    return model

def save_comparison_image(spectrogram, grad_mask, rl_mask, title, save_path):
    plt.figure(figsize=(10, 9))

    n_freq, n_frames = spectrogram.shape
    sample_rate = 16000
    hop_length = 256
    duration = (n_frames * hop_length) / sample_rate

    extent = [0, duration, 0, n_freq]

    plt.subplot(3, 1, 1)
    plt.title(f"[Original] {title}")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno', extent=extent)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel("Mel Frequency Bin")
    plt.xticks([])

    plt.subplot(3, 1, 2)
    plt.title("[Grad-CAM] Baseline")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=0.5, extent=extent)
    plt.imshow(grad_mask, aspect='auto', origin='lower', cmap='jet', alpha=0.6, extent=extent)
    plt.colorbar()
    plt.ylabel("Mel Frequency Bin")
    plt.xticks([])

    plt.subplot(3, 1, 3)
    plt.title("[RL-CAM] Ours (Optimized)")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=0.5, extent=extent)
    plt.imshow(rl_mask, aspect='auto', origin='lower', cmap='jet', alpha=0.6, extent=extent)
    plt.colorbar()
    plt.ylabel("Mel Frequency Bin")
    plt.xlabel("Time (seconds)")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def set_hybrid_mode(model):
    model.train()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
            m.eval()

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    model = load_model()
    ds = PatientGrbasDistDataset(csv_path=CSV_PATH)

    target_layer = model.encoder.cnn[2].net[0]
    grad_runner = GrbasGradCAM(model, target_layer)

    rl_runner = GrbasRLCAM(model)
    
    print(f"Starting Full Comparison for {len(ds)} patients...")
    
    for i in tqdm(range(len(ds)), desc="Processing"):
        try:
            sample = ds[i]
            pid = sample['patient_id']
            
            # 입력 준비
            input_tensor = sample['feat'].unsqueeze(0).to(DEVICE)
            input_tensor.requires_grad = True
            
            spectrogram = input_tensor[0, 0].detach().cpu().numpy()
            
            for item_idx, item_name in enumerate(ITEMS):

                set_hybrid_mode(model)
                grad_mask, pred_cls_idx = grad_runner(input_tensor, item_idx=item_idx)

                rl_mask, _ = rl_runner.generate(
                    input_tensor,
                    item_idx=item_idx,
                    target_class=pred_cls_idx,
                    steps=STEPS,
                    lr=LR,
                    lambda_reg=LAMBDA_REG
                )

                score = pred_cls_idx + 1
                filename = f"{pid}_{item_name}_Score{score}.png"
                save_path = os.path.join(SAVE_DIR, filename)
                
                save_comparison_image(
                    spectrogram,
                    grad_mask,
                    rl_mask,
                    title=f"Patient {pid} | Item {item_name} (Score: {score})",
                    save_path=save_path
                )
                
        except Exception as e:
            print(f"Error on patient {i}: {e}")
            continue

    print(f"Results saved in '{SAVE_DIR}'")

if __name__ == "__main__":
    main()