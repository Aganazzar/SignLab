import cv2
import time
import numpy as np
import torch
import pyttsx3

from realtime.feature_extractor import StreamingFeatureExtractor
from realtime.buffer import FeatureBuffer
from realtime.ctc_decoder import greedy_ctc_decode
from realtime.model import SignRecognitionModel
from sign_vocab import idx_to_sign, BLANK_IDX

# ---------------- CONFIG ---------------- #
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/sign_model.pth"
SEQUENCE_LENGTH = 45
SPEAK_COOLDOWN = 1.0  # seconds
MIN_CONFIDENCE = 0.7  # Minimum confidence threshold
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

# ---------------- TTS ---------------- #
tts = pyttsx3.init()
tts.setProperty("rate", 160)

def speak(text):
    tts.say(text)
    tts.runAndWait()

# ---------------- MODEL ---------------- #
def load_model(feature_dim):
    model = SignRecognitionModel(
        input_dim=feature_dim,
        hidden_dim=256,
        output_dim=len(idx_to_sign)
    ).to(DEVICE)

    if not torch.cuda.is_available():
        state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    else:
        state = torch.load(MODEL_PATH)

    model.load_state_dict(state)
    model.eval()
    return model

# ---------------- MAIN ---------------- #
def main():
    print("[INFO] Starting real-time sign recognition")
    extractor = StreamingFeatureExtractor()

    # Infer feature dimension
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feat = extractor.extract(dummy_frame)
    FEATURE_DIM = len(feat)
    print(f"[INFO] Feature dimension: {FEATURE_DIM}")


    buffer = FeatureBuffer(max_len=SEQUENCE_LENGTH, feature_dim=FEATURE_DIM)

    try:
        model = load_model(FEATURE_DIM)
        print("[INFO] Model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[ERROR] Make sure models/sign_model.pth exists and was trained properly.")
        return

    cap = cv2.VideoCapture(0)
    last_spoken = ""
    last_speak_time = 0
    
    # Boundary detection variables
    prev_features = None
    static_frame_count = 0
    is_signing = False
    accumulated_predictions = []
    accumulated_confidence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        features = extractor.extract(rgb)
        if features is not None:
            buffer.add(features)
            
            # Calculate hand velocity
            velocity = calculate_hand_velocity(features, prev_features)
            prev_features = features.copy()
            
            # Detect sign boundaries
            if velocity > VELOCITY_THRESHOLD:
                # Hand is moving - signing in progress
                is_signing = True
                static_frame_count = 0
            else:
                # Hand is static
                static_frame_count += 1

        # Run inference when buffer is full
        if buffer.is_full():
            seq = buffer.get()  # (T, F)
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)        # (1, T, C)
                probs = torch.softmax(logits, dim=-1)
                
                # Get confidence score
                max_probs = torch.max(probs, dim=-1)[0]
                confidence = max_probs.mean().item()
                
                preds = torch.argmax(probs, dim=-1).squeeze(0).cpu().numpy()

            decoded = greedy_ctc_decode(preds, blank=BLANK_IDX)
            text = " ".join(idx_to_sign[i] for i in decoded)
            
            # Accumulate predictions during signing
            if is_signing and text:
                accumulated_predictions.append(text)
                accumulated_confidence.append(confidence)
            
            # Output when hand becomes static after signing
            if is_signing and static_frame_count >= STATIC_FRAMES:
                if accumulated_predictions:
                    # Get most common prediction
                    from collections import Counter
                    most_common = Counter(accumulated_predictions).most_common(1)[0][0]
                    avg_confidence = np.mean(accumulated_confidence)
                    
                    now = time.time()
                    
                    # Filter: confidence + change detection + cooldown
                    if (avg_confidence >= MIN_CONFIDENCE and
                        most_common != last_spoken and 
                        now - last_speak_time > SPEAK_COOLDOWN):
                        
                        print(f"[SIGN] {most_common} (confidence: {avg_confidence:.2f})")
                        speak(most_common)
                        last_spoken = most_common
                        last_speak_time = now
                
                # Reset for next sign
                accumulated_predictions.clear()
                accumulated_confidence.clear()
                is_signing = False
                static_frame_count = 0

        cv2.imshow("Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
