#!/usr/bin/env python3
"""
Minimal UI - Exact copy of terminal version with simple display
"""
import streamlit as st
import cv2
import time
import numpy as np
import torch
from collections import deque
from PIL import Image

from realtime.feature_extractor import StreamingFeatureExtractor
from realtime.buffer import FeatureBuffer
from realtime.ctc_decoder import greedy_ctc_decode
from realtime.model import SignRecognitionModel
from sign_vocab import idx_to_sign, BLANK_IDX

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/sign_model.pth"
SEQUENCE_LENGTH = 45
SPEAK_COOLDOWN = 1.0
MIN_CONFIDENCE = 0.7  # Minimum confidence threshold
VELOCITY_THRESHOLD = 0.02  # Hand movement threshold for sign boundary
STATIC_FRAMES = 1  # Frames hand must be static to trigger output

def calculate_hand_velocity(features_current, features_prev):
    """Calculate hand movement velocity"""
    if features_prev is None:
        return 0.0
    hand_current = features_current[:63]
    hand_prev = features_prev[:63]
    velocity = np.linalg.norm(hand_current - hand_prev)
    return velocity

st.set_page_config(page_title="Sign Recognition", layout="wide")
st.title("Sign Language Recognition")

@st.cache_resource
def load_model(feature_dim):
    model = SignRecognitionModel(input_dim=feature_dim, hidden_dim=256, output_dim=len(idx_to_sign)).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
    model.load_state_dict(state)
    model.eval()
    return model

if 'running' not in st.session_state:
    st.session_state.running = False
if 'current_sign' not in st.session_state:
    st.session_state.current_sign = ""
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=10)

col1, col2 = st.columns([3, 1])

with col1:
    video_placeholder = st.empty()
    
with col2:
    st.subheader("Current Sign")
    sign_text = st.empty()
    st.subheader("History")
    history_text = st.empty()

col_start, col_stop = st.columns(2)
with col_start:
    if st.button("START", use_container_width=True):
        st.session_state.running = True
        st.rerun()

with col_stop:
    if st.button("STOP", use_container_width=True):
        st.session_state.running = False
        st.rerun()

if st.session_state.running:
    # Initialize exactly as terminal version
    extractor = StreamingFeatureExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feat = extractor.extract(dummy_frame)
    FEATURE_DIM = len(feat)
    
    buffer = FeatureBuffer(max_len=SEQUENCE_LENGTH, feature_dim=FEATURE_DIM)
    model = load_model(FEATURE_DIM)
    
    cap = cv2.VideoCapture(0)
    last_spoken = ""
    last_speak_time = 0
    
    # Boundary detection variables
    prev_features = None
    static_frame_count = 0
    is_signing = False
    accumulated_predictions = []
    accumulated_confidence = []
    
    frame_count = 0
    while st.session_state.running and frame_count < 500:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        features = extractor.extract(rgb)
        if features is not None:
            buffer.add(features)
            
            # Calculate velocity
            velocity = calculate_hand_velocity(features, prev_features)
            prev_features = features.copy()
            
            # Detect boundaries
            if velocity > VELOCITY_THRESHOLD:
                is_signing = True
                static_frame_count = 0
            else:
                static_frame_count += 1
        
        # Run inference when buffer is full
        if buffer.is_full():
            seq = buffer.get()
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                
                # Get confidence score
                max_probs = torch.max(probs, dim=-1)[0]
                confidence = max_probs.mean().item()
                
                preds = torch.argmax(probs, dim=-1).squeeze(0).cpu().numpy()
            
            decoded = greedy_ctc_decode(preds, blank=BLANK_IDX)
            text = " ".join(idx_to_sign[i] for i in decoded)
            
            # Accumulate during signing
            if is_signing and text:
                accumulated_predictions.append(text)
                accumulated_confidence.append(confidence)
            
            # Output when static
            if is_signing and static_frame_count >= STATIC_FRAMES:
                if accumulated_predictions:
                    from collections import Counter
                    most_common = Counter(accumulated_predictions).most_common(1)[0][0]
                    avg_confidence = np.mean(accumulated_confidence)
                    
                    now = time.time()
                    
                    if (avg_confidence >= MIN_CONFIDENCE and
                        most_common != last_spoken and 
                        now - last_speak_time > SPEAK_COOLDOWN):
                        
                        st.session_state.current_sign = most_common
                        st.session_state.history.append(most_common)
                        last_spoken = most_common
                        last_speak_time = now
                
                # Reset
                accumulated_predictions.clear()
                accumulated_confidence.clear()
                is_signing = False
                static_frame_count = 0
        
        cv2.putText(frame, f"Sign: {st.session_state.current_sign}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert BGR to RGB and then to PIL Image for Streamlit
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_frame)
        video_placeholder.image(pil_image, width=640)
        sign_text.markdown(f"## {st.session_state.current_sign.upper()}")
        history_text.write("\n".join([f"• {w}" for w in list(st.session_state.history)]))
        
        frame_count += 1
        if frame_count % 5 == 0:
            time.sleep(0.001)
    
    cap.release()
    if frame_count >= 500:
        st.session_state.running = False
        st.rerun()
    
if st.session_state.running:
    time.sleep(0.01)
    st.rerun()
