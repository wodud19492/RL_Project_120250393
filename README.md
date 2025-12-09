# RL_Project_120250393

    pip install -r requirements.txt

## Train (Regression)

    python train_v3.py --mode crnn_patient --epochs 500 --save_path best_crnn.pt --lr 5e-4

## Inference (Regression)

    python test_v3.py --mode crnn_patient --checkpoint best_crnn_5587.pt --check_baseline

## Train (RL-CAM)

    python cam_comp.py

## sample Inference

    python Inference.py

## PPT
[Download](https://github.com/wodud19492/RL-Project_120250393/blob/main/RL_Project_120250393.pptx)
