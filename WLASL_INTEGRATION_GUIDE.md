# WLASL Integration Guide

## 📖 Overview

**WLASL (Word-Level American Sign Language)** is a large-scale dataset with 2000+ signs. This guide shows how to add it to your project without breaking existing functionality.

## 🎯 What You'll Get

- **Expand vocabulary**: From 15 signs → 100-2000 signs
- **More training data**: Thousands of video samples
- **Keep existing system**: Your current 15-sign model stays intact
- **Separate models**: Train WLASL model separately

## 📋 Step-by-Step Process

### Step 1: Download WLASL Dataset

```bash
# Option A: Manual download
# 1. Go to: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
# 2. Click "Download"
# 3. Extract to: ./wlasl_raw/

# Option B: Using Kaggle API
pip install kaggle
# Configure API key (follow Kaggle instructions)
kaggle datasets download -d risangbaskoro/wlasl-processed
unzip wlasl-processed.zip -d wlasl_raw/
```

### Step 2: Process Videos to Features

```bash
# Process first 50 signs (recommended for testing)
conda activate sign_stream
python process_wlasl.py --input wlasl_raw --output wlasl_processed --max_signs 50

# Process 100 signs (takes ~1-2 hours)
python process_wlasl.py --max_signs 100

# Process all signs (takes several hours)
python process_wlasl.py --max_signs 2000
```

**What this does:**
- Reads video files from `wlasl_raw/`
- Extracts MediaPipe landmarks (same as your current system)
- Saves `.npy` feature files to `wlasl_processed/`
- Creates vocabulary file: `wlasl_processed/wlasl_vocab.json`

### Step 3: Train New Model

```bash
# Train on WLASL data (saves to models/wlasl_model.pth)
python train_wlasl.py --data wlasl_processed --epochs 50

# Your existing model remains at models/sign_model.pth
```

### Step 4: Use New Model (Optional)

To use the WLASL model in your UI:

1. Create `realtime_wlasl.py` (copy of `realtime_inference.py`)
2. Change:
   - `MODEL_PATH = "models/wlasl_model.pth"`
   - Load vocab from `models/wlasl_model_vocab.json`
3. Run: `python -m realtime.realtime_wlasl`

## 📊 Directory Structure After Processing

```
sign_lab/
├── dataset/                  # Your existing 15 signs (UNCHANGED)
│   ├── hello/
│   ├── yes/
│   └── ...
├── wlasl_raw/               # Downloaded videos (NEW)
│   ├── videos/
│   └── WLASL_v0.3.json
├── wlasl_processed/         # Processed features (NEW)
│   ├── hello/
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   ├── goodbye/
│   └── wlasl_vocab.json
├── models/
│   ├── sign_model.pth       # Your existing model (UNCHANGED)
│   ├── wlasl_model.pth      # New WLASL model
│   └── wlasl_model_vocab.json
```

## ⚙️ Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_signs` | 100 | Limit number of signs (start small!) |
| `--min_samples` | 5 | Minimum videos per sign |
| `--sequence_length` | 45 | Frames per sequence (matches current) |

## ⏱️ Expected Processing Time

- **50 signs**: ~30 minutes
- **100 signs**: ~1-2 hours  
- **500 signs**: ~5-8 hours
- **2000 signs**: ~15-20 hours

## 🚨 Important Notes

1. **Start Small**: Begin with 50-100 signs to test
2. **Storage**: Each sign needs ~1-5 MB per sample
3. **Memory**: Processing uses ~2-4 GB RAM
4. **Your existing system stays intact**: All current files unchanged
5. **GPU/MPS**: Will use Apple Silicon MPS if available

## 🔍 Troubleshooting

**"Input directory not found"**
```bash
# Make sure you downloaded and extracted WLASL first
ls wlasl_raw/  # Should show videos/ folder
```

**"No videos found"**
```bash
# Check structure
ls wlasl_raw/videos/*.mp4  # Should show video files
```

**Processing too slow**
```bash
# Reduce number of signs
python process_wlasl.py --max_signs 20
```

## 📈 Next Steps After Training

1. **Evaluate**: Check accuracy in train_wlasl.py output
2. **Compare**: Test against your existing 15-sign model
3. **Integrate**: Modify UI to support model switching
4. **Expand**: Gradually increase number of signs

## 💡 Tips

- Start with 50 signs that are most useful to you
- Check which signs have most samples in vocabulary
- Your current 15-sign system keeps working during this process
- You can run both models side-by-side
