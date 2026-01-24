import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import string  


# Load trained model
model = tf.keras.models.load_model("saved_model/hand_sign_cnn_model.h5")


LABELS = ["A", "B", "C"] 

# 2. Initialize MediaPipe

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# 3. Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to access webcam.")
        break

    # Flip image for mirror view and convert to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    predicted_letter = None
    predicted_confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = image.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            xmin, xmax = int(min(x_coords)) - 40, int(max(x_coords)) + 40
            ymin, ymax = int(min(y_coords)) - 40, int(max(y_coords)) + 40
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            hand_img = image[ymin:ymax, xmin:xmax]

            if hand_img.size > 0:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_img = cv2.resize(hand_img, (128, 128))
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                prediction = model.predict(hand_img, verbose=0)
                pred_index = np.argmax(prediction)
                confidence = np.max(prediction)
                pred_label = LABELS[pred_index]

                # Show if confidence high enough
                if confidence > 0.2:
                    predicted_letter = pred_label
                    predicted_confidence = confidence

                    y_text = max(30, ymin - 10)
                    cv2.putText(image, f"{pred_label} ({confidence*100:.1f}%)",
                                (xmin, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # -----------------------------
    # Display current prediction at top-left
    # -----------------------------
    if predicted_letter:
        cv2.rectangle(image, (0, 0), (300, 70), (0, 0, 0), -1)
        cv2.putText(image,
                    f"Letter: {predicted_letter}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image,
                    f"Accuracy: {predicted_confidence*100:.1f}%",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show webcam window
    cv2.imshow("Hand Gesture Detection (A–Z)", image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break
# 4. Cleanup
cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import string  # for A–Z labels


# 1. Load trained model
model = tf.keras.models.load_model("saved_model/hand_sign_cnn_model.h5")

# ⚠️ Make sure these match your training labels
LABELS = ["A", "B", "C"]  # or list(string.ascii_uppercase) if you trained A–Z

# -----------------------------
# 2. Initialize MediaPipe
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# 3. Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to access webcam.")
        break

    # Flip image for mirror view and convert to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    predicted_letter = None
    predicted_confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = image.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            xmin, xmax = int(min(x_coords)) - 40, int(max(x_coords)) + 40
            ymin, ymax = int(min(y_coords)) - 40, int(max(y_coords)) + 40
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            hand_img = image[ymin:ymax, xmin:xmax]

            if hand_img.size > 0:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_img = cv2.resize(hand_img, (128, 128))
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                prediction = model.predict(hand_img, verbose=0)
                pred_index = np.argmax(prediction)
                confidence = np.max(prediction)
                pred_label = LABELS[pred_index]

                # Show if confidence high enough
                if confidence > 0.2:
                    predicted_letter = pred_label
                    predicted_confidence = confidence

                    y_text = max(30, ymin - 10)
                    cv2.putText(image, f"{pred_label} ({confidence*100:.1f}%)",
                                (xmin, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display current prediction at top-left
    if predicted_letter:
        cv2.rectangle(image, (0, 0), (300, 70), (0, 0, 0), -1)
        cv2.putText(image,
                    f"Letter: {predicted_letter}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image,
                    f"Accuracy: {predicted_confidence*100:.1f}%",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show webcam window
    cv2.imshow("Hand Gesture Detection (A–Z)", image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
