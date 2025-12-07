import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# ==========================================
# 1. Constants & Configuration
# ==========================================
VOWEL_MAP = {
    "001": 0,  # /a/
    "007": 1,  # /i/
    "008": 2,  # /u/
}
ITEMS = ["G", "R", "B", "A", "S"]


# ==========================================
# 2. Utility Functions (Label & Dist Processing)
# ==========================================
def parse_rating(val):
    """CSV 값을 안전하게 1.0 ~ 5.0 사이의 float로 변환합니다."""
    if pd.isna(val):
        return None
    
    if isinstance(val, str):
        v = val.strip()
        if v in ["", "-", "NaN", "nan"]:
            return None
        try:
            val = float(v)
        except ValueError:
            return None
    else:
        val = float(val)

    if val < 1 or val > 5:
        return None
    return val

def build_distribution_from_row(row, items=ITEMS):
    """
    한 행(row)에서 SLP1, SLP2, SLP3 데이터를 읽어 (5, 5) 분포 텐서를 생성합니다.
    Returns: (5, 5) tensor -> (Items, Classes[0-4])
    """
    dist = torch.zeros(len(items), 5, dtype=torch.float32)

    for i, item in enumerate(items):
        # 1) 개별 평가자 점수 수집
        raw_ratings = [row.get(f"SLP{k}_{item}") for k in [1, 2, 3]]
        ratings = [parse_rating(r) for r in raw_ratings if parse_rating(r) is not None]

        # 2) 평가자 점수가 없으면 평균값(SLPall) 사용
        if not ratings:
            mean_val = parse_rating(row.get(f"SLPall_{item}"))
            if mean_val is not None:
                idx = int(round(mean_val))
                idx = max(1, min(5, idx)) - 1  # 1~5 -> 0~4
                dist[i, idx] = 1.0
            else:
                # 최후의 수단: Uniform distribution
                dist[i] = torch.ones(5) / 5.0
            continue

        # 3) 평가자 점수로 분포 생성
        for v in ratings:
            idx = int(round(v))
            idx = max(1, min(5, idx)) - 1
            dist[i, idx] += 1.0

        dist[i] /= len(ratings) # 정규화 (합이 1이 되도록)

    return dist


# ==========================================
# 3. Datasets
# ==========================================
class BaseGrbasDataset(Dataset):
    """
    모든 GRBAS 데이터셋의 공통 기능을 담당하는 부모 클래스입니다.
    오디오 로딩, 전처리, Mel-Spectrogram 변환을 담당합니다.
    """
    def __init__(self, csv_path, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=256, duration=20.0):
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * duration)
        
        # Audio Transforms
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def _load_wav(self, path):
        """Wav 로드, 리샘플링, 모노 변환, 패딩/자르기"""
        wav, sr = torchaudio.load(path)
        
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True) # Mono

        current_samples = wav.shape[1]
        if current_samples > self.target_samples:
            wav = wav[:, :self.target_samples]
        elif current_samples < self.target_samples:
            pad_amount = self.target_samples - current_samples
            wav = F.pad(wav, (0, pad_amount))

        return wav  # (1, T)

    def _wav_to_logmel(self, wav):
        """Wav -> Log Mel Spectrogram"""
        with torch.no_grad():
            mel = self.melspec(wav)
            mel_db = self.to_db(mel)
        return mel_db # (1, n_mels, T')

    def __getitem__(self, idx):
        raise NotImplementedError


class GrbasDataset(BaseGrbasDataset):
    """기본 Regression용 Dataset (Target: 평균 점수)"""
    def __init__(self, normalize_labels=False, **kwargs):
        super().__init__(**kwargs)
        self.normalize_labels = normalize_labels
        self.y_max = 5.0 if normalize_labels else 1.0

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Audio Processing
        wav = self._load_wav(row["wav_path"])
        feat = self._wav_to_logmel(wav)

        # Vowel ID
        vowel_label = str(row["vowel_label"]).zfill(3)
        vowel_id = VOWEL_MAP.get(vowel_label, 0) # Default to 0 if not found

        # Target (Mean Scores)
        y = torch.tensor([row[f"SLPall_{item}"] for item in ITEMS], dtype=torch.float32)
        
        if self.normalize_labels:
            y = y / self.y_max

        return {
            "feat": feat,
            "label": y,
            "vowel_id": torch.tensor(vowel_id, dtype=torch.long),
            "patient_id": row["patient_ID"],
        }


class GrbasDistDataset(BaseGrbasDataset):
    """Distribution Learning용 Dataset (Target: 분포 & 평균)"""
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Audio
        wav = self._load_wav(row["wav_path"])
        feat = self._wav_to_logmel(wav) # (1, F, T)

        # Targets
        target_dist = build_distribution_from_row(row)
        target_mean = torch.tensor([row[f"SLPall_{item}"] for item in ITEMS], dtype=torch.float32)

        return {
            "feat": feat,
            "target_dist": target_dist,
            "target_mean": target_mean,
            "patient_id": row["patient_ID"],
            "vowel_label": str(row["vowel_label"]),
        }


class PatientGrbasDistDataset(BaseGrbasDataset):
    """환자 단위 3모음(/a, /i, /u) Joint Dataset"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Grouping Logic
        self.groups = []
        grouped = self.df.groupby("patient_ID")

        for pid, gdf in grouped:
            sub = {}
            valid_group = True
            for v_code in VOWEL_MAP:
                # 해당 모음 코드를 가진 행 찾기
                rows = gdf[gdf["vowel_label"].astype(str).str.zfill(3) == v_code]
                if len(rows) == 0:
                    valid_group = False
                    break
                sub[v_code] = rows.iloc[0] # 첫 번째 행 사용

            if valid_group:
                self.groups.append({"patient_id": pid, "rows": sub})

        print(f"[PatientGrbasDistDataset] Usable patients: {len(self.groups)}")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        rows = group["rows"]
        
        feats = []
        # /a, /i, /u 순서대로 feature 추출 및 적재
        for v_code in VOWEL_MAP:
            row = rows[v_code]
            wav = self._load_wav(row["wav_path"])
            mel = self._wav_to_logmel(wav)
            feats.append(mel)

        # (3, F, T) 형태로 결합
        feat_3 = torch.cat(feats, dim=0)

        # 라벨은 임의의 행(여기선 첫번째)에서 추출 (환자 단위로 동일하다고 가정)
        any_row = list(rows.values())[0]
        target_dist = build_distribution_from_row(any_row)
        target_mean = torch.tensor([any_row[f"SLPall_{item}"] for item in ITEMS], dtype=torch.float32)

        return {
            "feat": feat_3,
            "target_dist": target_dist,
            "target_mean": target_mean,
            "patient_id": group["patient_id"],
        }


# ==========================================
# 4. Models
# ==========================================
class ConvBlock(nn.Module):
    """Standard 2D Conv Block"""
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), pool_size=(2, 2), padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size)
        )
    
    def forward(self, x):
        return self.net(x)

class GrbasCNN(nn.Module):
    """Basic CNN Model with Vowel Embedding"""
    def __init__(self, num_vowels=3, use_vowel_embedding=True):
        super().__init__()
        self.use_vowel_embedding = use_vowel_embedding

        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        in_fc = 128
        if use_vowel_embedding:
            self.vowel_emb = nn.Embedding(num_vowels, 8)
            in_fc += 8

        self.classifier = nn.Sequential(
            nn.Linear(in_fc, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),    nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 5) # G, R, B, A, S
        )

    def forward(self, feat, vowel_id=None):
        x = self.features(feat)          # (B, 128, H, W)
        x = self.global_pool(x).flatten(1) # (B, 128)

        if self.use_vowel_embedding:
            v = self.vowel_emb(vowel_id)
            x = torch.cat([x, v], dim=-1)

        return self.classifier(x)


class CRNNEncoder(nn.Module):
    """CRNN Encoder used in AutoGRBAS"""
    def __init__(self, in_channels=1, conv_channels=(32, 64, 128), 
                 gru_hidden=128, gru_layers=2, dropout=0.3):
        super().__init__()
        
        # CNN
        ch1, ch2, ch3 = conv_channels
        self.cnn = nn.Sequential(
            ConvBlock(in_channels, ch1),
            ConvBlock(ch1, ch2),
            ConvBlock(ch2, ch3), # F, T both reduced
            nn.Dropout(dropout)
        )
        
        # RNN Setup
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.proj_size = 256
        
        self.gru = nn.GRU(
            input_size=self.proj_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.proj_conv = None # Lazy init

    def _get_proj_layer(self, cf_dim, device):
        if self.proj_conv is None:
            self.proj_conv = nn.Conv1d(cf_dim, self.proj_size, kernel_size=1).to(device)
        return self.proj_conv

    def forward(self, x):
        # x: (B, C, F, T)
        x = self.cnn(x)
        B, C, Freq, Time = x.size()

        # Reshape for RNN: (B, C*F, T)
        x = x.view(B, C * Freq, Time)
        
        # Projection (C*F -> proj_size)
        proj_layer = self._get_proj_layer(C * Freq, x.device)
        x = self.dropout(F.relu(proj_layer(x)))

        # Permute for GRU: (B, T, proj_size)
        x = x.permute(0, 2, 1)

        # GRU
        output, _ = self.gru(x)
        
        # Mean Pooling over time
        enc = output.mean(dim=1) # (B, hidden * 2)
        return enc


class AutoGRBASModel(nn.Module):
    """Auto-GRBAS Distribution Learning Model"""
    def __init__(self, in_channels=1, gru_hidden=128, dropout=0.3):
        super().__init__()
        
        self.encoder = CRNNEncoder(in_channels=in_channels, gru_hidden=gru_hidden, dropout=dropout)
        
        enc_dim = gru_hidden * 2 # Bidirectional
        
        self.head = nn.Sequential(
            nn.Linear(enc_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 5 * 5) # 5 Items * 5 Scores
        )
        
        # Score values [1, 2, 3, 4, 5] for expected value calculation
        self.register_buffer("score_values", torch.arange(1, 6, dtype=torch.float32))

    def forward(self, x, return_scores=False):
        enc = self.encoder(x)
        logits = self.head(enc).view(-1, 5, 5) # (B, 5, 5)

        if not return_scores:
            return logits
            
        # Calculate Expected Scores
        probs = F.softmax(logits, dim=-1)
        scores = torch.sum(probs * self.score_values.view(1, 1, -1), dim=-1) # (B, 5)
        
        return logits, scores


# ==========================================
# 5. Losses
# ==========================================
def grbas_distribution_loss(logits, target_dist):
    """
    KL Divergence-like loss using Cross Entropy with soft targets
    logits: (B, 5, 5)
    target_dist: (B, 5, 5)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # - sum(target * log(pred))
    loss = -(target_dist * log_probs).sum(dim=-1).mean()
    return loss