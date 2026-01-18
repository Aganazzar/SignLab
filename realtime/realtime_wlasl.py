import cv2
import time
import json
import numpy as np
import torch
import pyttsx3
from pathlib import Path

from realtime.feature_extractor import StreamingFeatureExtractor
from realtime.buffer import FeatureBuffer
from realtime.ctc_decoder import greedy_ctc_decode
from realtime.model import SignRecognitionModel

# ---------------- CONFIG ---------------- #
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/wlasl_model.pth"
VOCAB_PATH = "models/wlasl_model_vocab.json"
SEQUENCE_LENGTH = 45
SPEAK_COOLDOWN = 1.0  # seconds
MIN_CONFIDENCE = 0.5  # Minimum confidence threshold
VELOCITY_THRESHOLD = 0.02  # Hand movement threshold for sign boundary
STATIC_FRAMES = 1  # Frames hand must be static to trigger output
# --------------------------------------- #

# ---------------- BOUNDARY DETECTION ---------------- #
def calculate_hand_velocity(features_current, features_prev):
    """
    Calculate hand movement velocity between two frames.
    Uses first 21*3 = 63 features (right hand landmarks x,y,z)
    """
    if features_prev is None:
        return 0.0
    
    # Extract hand landmarks (first 63 features = 21 landmarks * 3 coords)
    hand_current = features_current[:63]
    hand_prev = features_prev[:63]
    
    # Calculate Euclidean distance
    velocity = np.linalg.norm(hand_current - hand_prev)
    return velocity

# ---------------- LOAD VOCABULARY ---------------- #
def load_vocabulary(vocab_path):
    """
    Load vocabulary from JSON file.
    
    Args:
        vocab_path: Path to vocab JSON file (e.g., models/wlasl_model_vocab.json)
    
    Returns:
        idx_to_sign: dict mapping index -> sign name
        sign_to_idx: dict mapping sign name -> index
        BLANK_IDX: index of blank token
    """
    with open(vocab_path, 'r') as f:
        sign_to_idx = json.load(f)
    
    # Create reverse mapping (index -> sign)
    idx_to_sign = {idx: sign for sign, idx in sign_to_idx.items()}
    
    # Add blank token at the end
    num_signs = len(idx_to_sign)
    BLANK_IDX = num_signs
    idx_to_sign[BLANK_IDX] = "<blank>"
    
    print(f"[INFO] Loaded {num_signs} signs from vocabulary")
    print(f"[INFO] Sample signs: {list(sign_to_idx.keys())[:10]}")
    
    return idx_to_sign, sign_to_idx, BLANK_IDX

# Load vocabulary
idx_to_sign, sign_to_idx, BLANK_IDX = load_vocabulary(VOCAB_PATH)

# ---------------- TTS ---------------- #
tts = pyttsx3.init()
tts.setProperty("rate", 160)

def speak(text):
    tts.say(text)
    tts.runAndWait()

# ---------------- MODEL ---------------- #
def load_model(feature_dim):
    # Load model with correct dimensions from training
    # Training used 1548 features (some samples had missing landmarks)
    # We need to match that dimension
    
    # First load the state to get actual dimensions
    if not torch.cuda.is_available():
        state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    else:
        state = torch.load(MODEL_PATH)
    
    # Get input dimension from saved weights
    trained_input_dim = state['gru.weight_ih_l0'].shape[1]
    trained_output_dim = state['fc.weight'].shape[0]
    
    print(f"[INFO] Model was trained with {trained_input_dim} input features and {trained_output_dim} output classes")
    
    model = SignRecognitionModel(
        input_dim=trained_input_dim,
        hidden_dim=256,
        output_dim=trained_output_dim
    ).to(DEVICE)

    model.load_state_dict(state)
    model.eval()
    return model, trained_input_dim

# ---------------- MAIN ---------------- #
def main():
    print("[INFO] Starting WLASL real-time sign recognition")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Model: {MODEL_PATH}")
    print(f"[INFO] Vocabulary: {len(idx_to_sign)} signs")
    
    extractor = StreamingFeatureExtractor()

    # Infer feature dimension
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feat = extractor.extract(dummy_frame)
    FEATURE_DIM = len(feat)

    # Create buffer + load model
    model, trained_feature_dim = load_model(FEATURE_DIM)
    
    # Use trained dimension for buffer
    buffer = FeatureBuffer(max_len=SEQUENCE_LENGTH, feature_dim=trained_feature_dim)
    
    print(f"[INFO] Live feature dimension: {FEATURE_DIM}")
    print(f"[INFO] Model expects: {trained_feature_dim}")
    
    if FEATURE_DIM != trained_feature_dim:
        print(f"[WARNING] Feature dimension mismatch! Will truncate/pad to {trained_feature_dim}")

    cap = cv2.VideoCapture(0)
    last_spoken = ""
    last_speak_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract features
        features = extractor.extract(rgb)
        if features is not None:
            # Adjust feature dimension to match model
            if len(features) > trained_feature_dim:
                features = features[:trained_feature_dim]  # Truncate
            elif len(features) < trained_feature_dim:
                # Pad with zeros
                padding = np.zeros(trained_feature_dim - len(features))
                features = np.concatenate([features, padding])
            
            buffer.add(features)

        # When buffer is full → predict
        if buffer.is_full():
            seq = buffer.get()  # (SEQUENCE_LENGTH, FEATURE_DIM)
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                
                # Get confidence score
                max_probs = torch.max(probs, dim=-1)[0]
                confidence = max_probs.mean().item()
                
                preds = torch.argmax(probs, dim=-1).squeeze(0).cpu().numpy()

            # CTC decode
            decoded = greedy_ctc_decode(preds, blank=BLANK_IDX)
            text = " ".join(idx_to_sign[i] for i in decoded)

            # Filter: confidence + change detection + cooldown
            now = time.time()
            if (text and 
                confidence >= MIN_CONFIDENCE and
                text != last_spoken and 
                now - last_speak_time > SPEAK_COOLDOWN):
                
                print(f"[RECOGNIZED] {text} (confidence: {confidence:.2f})")
                speak(text)
                last_spoken = text
                last_speak_time = now

        # Display
        cv2.putText(frame, f"Sign: {last_spoken}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("WLASL Sign Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
