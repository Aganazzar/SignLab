"""
Collect training data for sign language recognition.
Records hand landmark sequences for each sign with visualization.

WORKFLOW:
1. Choose which signs to collect at startup (or collect all)
2. Shows all available signs to collect
3. For each sign, shows hand landmarks in real-time
4. You perform the sign, press ENTER to start, ESC to stop
5. Records hand position + movements
6. Saves to dataset/{sign_name}/{sample_num}.npy
"""
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core import base_options

from realtime.feature_extractor import StreamingFeatureExtractor, USE_FACE
from sign_vocab import sign_to_idx

# Config
DATASET_DIR = "dataset"
SEQUENCE_LENGTH = 45
SAMPLES_PER_SIGN = 50  # Change this to collect more/fewer samples
os.makedirs(DATASET_DIR, exist_ok=True)

# Initialize MediaPipe landmarkers
options = HandLandmarkerOptions(
    base_options=base_options.BaseOptions(
        model_asset_path="mediapipe/hand_landmarker.task"
    ),
    num_hands=2,
    min_hand_detection_confidence=0.5
)
hand_landmarker = HandLandmarker.create_from_options(options)

# Face landmarker
face_landmarker = None
if USE_FACE:
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    try:
        face_options = FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(
                model_asset_path="mediapipe/face_landmarker.task"
            ),
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        face_landmarker = FaceLandmarker.create_from_options(face_options)
<<<<<<< HEAD
        print("✓ Face detection enabled\n")
    except Exception as e:
        print(f"⚠️  Face detection disabled: {e}")
=======
        print("Face detection enabled\n")
    except Exception as e:
        print(f"Face detection disabled: {e}")
>>>>>>> 2dee0b8 (model.pth added for reference)
        print("   Run: python setup_models.py\n")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

extractor = StreamingFeatureExtractor()

def draw_hand_landmarks(frame, detection_result):
    """Draw hand skeleton on frame"""
    h, w, _ = frame.shape
    
    if detection_result.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Draw landmarks (circles)
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            
            # Draw connections (skeleton)
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame

def draw_face_landmarks(frame, face_result):
    """Draw key face landmarks"""
    if not face_result or not face_result.face_landmarks:
        return frame
    
    h, w, _ = frame.shape
    face_lm = face_result.face_landmarks[0]
    
    # Draw key facial points
    key_indices = [
        33, 133, 362, 263,  # Eyes
        61, 291, 13, 14,     # Mouth
        1, 152,              # Nose, chin
        10, 234, 454,        # Forehead, cheeks
    ]
    
    for idx in key_indices:
        if idx < len(face_lm):
            x = int(face_lm[idx].x * w)
            y = int(face_lm[idx].y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)
    
    # Draw face oval
    for connection in mp_face_mesh.FACEMESH_FACE_OVAL:
        start_idx = connection[0]
        end_idx = connection[1]
        
        if start_idx < len(face_lm) and end_idx < len(face_lm):
            start = face_lm[start_idx]
            end = face_lm[end_idx]
            
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
    
    return frame

def collect_sign(sign_name):
    """Collect training samples for a single sign automatically frame by frame"""
    global hand_landmarker, face_landmarker
    sign_dir = os.path.join(DATASET_DIR, sign_name)
    os.makedirs(sign_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    sample_count = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
    
    print(f"\n{'='*60}")
    print(f"COLLECTING: {sign_name.upper()}")
    print(f"{'='*60}")
    print(f"Progress: {sample_count}/{SAMPLES_PER_SIGN} samples collected\n")
    print("  → Position your hands in frame")
    print("  → Press ENTER to start collecting this sign")
    print("  → Press ESC at any time to stop and move to next sign\n")
    
    # Wait for ENTER to start the sign collection
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Sign: {sign_name.upper()}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
        cv2.putText(frame, "Press ENTER to start", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imshow("Data Collection - Hand Landmarks Visible", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 13:  # ENTER
            break
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            print(f"Skipped {sign_name}")
            return
    
    # Start automatic sample collection
    while sample_count < SAMPLES_PER_SIGN:
        sequence = []
        print(f"Sample {sample_count + 1}/{SAMPLES_PER_SIGN} - Recording {SEQUENCE_LENGTH} frames")
        
        frames_recorded = 0
        while frames_recorded < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_np = np.array(rgb, dtype=np.uint8)
            
            # Detect hands
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)
            hand_result = hand_landmarker.detect(mp_image)
            
            # Detect face
            face_result = None
            if face_landmarker:
                try:
                    face_result = face_landmarker.detect(mp_image)
                except:
                    pass
            
            # Extract features
            features = extractor.extract(rgb_np)
            if features is not None:
                # Ensure consistent size by padding to expected dimension
                expected_size = extractor.total_feature_dim
                if len(features) < expected_size:
                    features = np.concatenate([features, np.zeros(expected_size - len(features))])
                elif len(features) > expected_size:
                    features = features[:expected_size]
                sequence.append(features)
            
            frames_recorded += 1
            
            # Draw landmarks
            frame = draw_hand_landmarks(frame, hand_result)
            if face_result:
                frame = draw_face_landmarks(frame, face_result)
            
            # Status display
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (w-10, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Sign: {sign_name.upper()}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
            cv2.putText(frame, f"Recording Sample {sample_count + 1}/{SAMPLES_PER_SIGN} ({frames_recorded}/{SEQUENCE_LENGTH} frames)", 
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.imshow("Data Collection - Hand Landmarks Visible", frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC pressed: stop current sign immediately
                cap.release()
                cv2.destroyAllWindows()
                print(f"Stopped {sign_name} early at sample {sample_count + 1}")
                return
        
        # Save sequence
        save_path = os.path.join(sign_dir, f"{sample_count}.npy")
        np.save(save_path, np.array(sequence))
        sample_count += 1
        print(f"Saved Sample {sample_count} ({SEQUENCE_LENGTH} frames)")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished collecting {sign_name}")
    print(f"{'='*60}\n")
    print("Press ENTER to start collecting next sign...")


def main():
    print("\n" + "="*60)
    print("  SIGN LANGUAGE DATA COLLECTION")
    print("="*60)
    print()
    
    # Get all available signs
    all_signs = [k for k in sign_to_idx.keys() if k != "<blank>"]
    
    print("Available signs:")
    for i, sign in enumerate(all_signs, 1):
        print(f"  {i}. {sign}")
    
    print()
    print("Choose which signs to collect:")
    print("  Enter numbers separated by spaces (e.g.: 1 2 3)")
    print("  Or press ENTER to collect ALL signs")
    print()
    
    choice = input("Your choice: ").strip()
    
    if choice == "":
        # Collect all signs
        SIGNS = all_signs
        print(f"\nWill collect all {len(SIGNS)} signs\n")
    else:
        # Parse user selection
        try:
            indices = [int(x) - 1 for x in choice.split()]
            SIGNS = [all_signs[i] for i in indices if 0 <= i < len(all_signs)]
            if not SIGNS:
                print("Invalid selection, collecting all signs instead")
                SIGNS = all_signs
            else:
                print(f"\nSelected signs: {', '.join(SIGNS)}\n")
        except (ValueError, IndexError):
            print("Invalid input, collecting all signs instead")
            SIGNS = all_signs
    
    print(f"Total signs to collect: {len(SIGNS)}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print()
    
    # Show status
    print("Status:")
    for idx, sign in enumerate(SIGNS, 1):
        sign_dir = os.path.join(DATASET_DIR, sign)
        existing = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')]) if os.path.exists(sign_dir) else 0
        
        if existing >= SAMPLES_PER_SIGN:
            print(f"  [{idx}/{len(SIGNS)}] {sign:12s} - {existing}/{SAMPLES_PER_SIGN} (complete)")
        else:
            print(f"    [{idx}/{len(SIGNS)}] {sign:12s} - {existing}/{SAMPLES_PER_SIGN} (need to collect)")
    
    print("\n" + "="*60 + "\n")
    
    # Collect data
    for idx, sign in enumerate(SIGNS, 1):
        sign_dir = os.path.join(DATASET_DIR, sign)
        existing = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')]) if os.path.exists(sign_dir) else 0
        
        if existing >= SAMPLES_PER_SIGN:
            continue
        
        print(f"\nNEXT: Sign [{idx}/{len(SIGNS)}] - {sign.upper()}")
        collect_sign(sign)
    
    print("\n" + "="*60)
    print("Data collection complete!")
    print()
    print("NEXT STEP:")
    print("  python train.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
