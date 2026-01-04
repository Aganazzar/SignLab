# Real-Time Sign Language Recognition

A minimal PyTorch-based real-time sign language interpreter using MediaPipe hand tracking.

## Quick Start

### 1. Collect Training Data
```bash
python collect_data.py
```
- Shows each sign to collect
- Press SPACE to start recording each sample
- ESC to skip a sign
- Collects 50 samples per sign (configurable)

### 2. Train the Model
```bash
python train.py
```
- Trains on collected data
- Automatically saves best model
- ~50 epochs

### 3. Run Real-Time Recognition
```bash
python -m realtime.realtime_inference
```
- Activates webcam
- Shows live predictions
- Press ESC to quit

## Structure
```
sign_lab/
├── realtime/              # Core inference module
│   ├── realtime_inference.py  # Main entry point
│   ├── feature_extractor.py   # MediaPipe feature extraction
│   ├── buffer.py              # Sequence buffering
│   ├── model.py               # PyTorch model
│   └── ctc_decoder.py         # CTC decoding
├── dataset/               # Training data (auto-created)
├── models/
│   └── sign_model.pth     # Trained model
├── collect_data.py        # Data collection script
├── train.py               # Training script
└── sign_vocab.py          # Sign vocabulary
```

## Supported Signs
hello, thank_you, yes, no, please, sorry, i, you, help

## Configuration
Edit at top of each script:
- `SAMPLES_PER_SIGN`: How many training samples per sign (default: 50)
- `SEQUENCE_LENGTH`: Frames per gesture (default: 45)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Training epochs (default: 50)
