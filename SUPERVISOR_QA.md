# Supervisor Q&A: Real-Time Sign Language Recognition

Q1. What problem does the project solve and what is the end-to-end flow?
A1. It provides a minimal real-time sign language interpreter: (1) collect labeled gesture sequences from a webcam, (2) train a sequence model on those samples, and (3) run live inference with text-to-speech output. Data is captured by [collect_data.py](collect_data.py), trained via [train.py](train.py), and served by [realtime/realtime_inference.py](realtime/realtime_inference.py).

Q2. What are the core dependencies and runtime requirements?
A2. Python with PyTorch for modeling, MediaPipe Tasks for landmark detection (hand and optional face), OpenCV for webcam capture, NumPy for tensor I/O, and pyttsx3 for TTS. The code auto-selects Metal Performance Shaders on macOS, CUDA on other GPUs, else CPU. Model checkpoints live under models/.

Q3. How is data collected and stored?
A3. [collect_data.py](collect_data.py) streams webcam frames, extracts hand landmarks, and records fixed-length sequences (default 45 frames) when the user presses SPACE. Each sample is saved as a .npy array under dataset/<sign_name>/, one folder per sign. Default target is 50 samples per sign (configurable at the top of the script).

Q4. How is the vocabulary defined and where are labels mapped?
A4. Vocabulary and integer IDs are defined in [sign_vocab.py](sign_vocab.py). A blank token is reserved for CTC-style decoding. The file writes a JSON snapshot to sign_vocab.json for reference. Update the SIGNS/idx_to_sign map and regenerate data before retraining when adding or removing signs.

Q5. What features are extracted from each frame?
A5. [realtime/feature_extractor.py](realtime/feature_extractor.py) uses MediaPipe to pull up to two hands (21 landmarks each), normalizes them to be wrist-centered and scale-invariant, and computes: raw landmark coordinates, all pairwise distances, four joint angles, optional face keypoint positions (selected 68 indices), simple facial metrics (eye/mouth spans, symmetry, jaw width), optional head pose deltas, and per-feature velocities by differencing consecutive frames. All features are concatenated into a single vector per frame; velocity doubles the base dimension.

Q6. How are sequences handled for training?
A6. [train.py](train.py) loads every .npy sample per sign, pads with zeros or truncates to the fixed sequence length (default 45 frames), and ensures feature dimensions align with the extractor’s expected width. Samples are split 80/20 into train/test subsets via torch.utils.data.random_split.

Q7. What is the model architecture?
A7. [realtime/model.py](realtime/model.py) defines a bidirectional GRU (2 layers, hidden size 256) followed by a linear classifier and log-softmax. Input shape is (batch, time, feature_dim); output is per-frame class log probabilities over the vocabulary.

Q8. What loss and optimization strategy are used?
A8. Training repeats the ground-truth label across all frames of a sequence and applies CrossEntropyLoss to per-frame logits (flattened). Optimizer is Adam with learning rate 1e-3 and StepLR scheduler (step_size=10, gamma=0.5). Gradients are clipped to a norm of 1.0.

Q9. How is the best checkpoint selected and saved?
A9. After each epoch, validation loss and sequence accuracy (mode of per-frame predictions) are computed. The lowest validation loss so far triggers a save to models/sign_model.pth. Training defaults: batch size 32, epochs 50, sequence length 45.

Q10. How does real-time inference work?
A10. [realtime/realtime_inference.py](realtime/realtime_inference.py) captures webcam frames, extracts features, and pushes them into a fixed-length FeatureBuffer. Once the buffer holds SEQUENCE_LENGTH frames, the stacked tensor is fed to the loaded GRU model. Argmax over classes per frame is decoded with a greedy CTC decoder ([realtime/ctc_decoder.py](realtime/ctc_decoder.py)) to handle repeated blanks. Detected text is spoken via pyttsx3 with a cooldown to avoid repeats.

Q11. What is the role of the buffer and decoder?
A11. The buffer ([realtime/buffer.py](realtime/buffer.py)) enforces a sliding window of exactly 45 frames, providing temporal context without unbounded growth. The greedy CTC decoder collapses repeats and removes blanks, turning per-frame class IDs into a concise sign string.

Q12. How do you add a new sign to the system?
A12. Add the sign name to the mappings in [sign_vocab.py](sign_vocab.py), regenerate sign_vocab.json by rerunning the script, run [collect_data.py](collect_data.py) to record new samples into dataset/<new_sign>/, then retrain with [train.py](train.py). Ensure you collect enough samples (50+ recommended) for the new class.

Q13. How is feature dimension consistency enforced between training and inference?
A13. During training, [train.py](train.py) checks each loaded sequence against StreamingFeatureExtractor.total_feature_dim and pads or truncates the feature axis to match. During inference, the feature dimension is inferred from a dummy frame via the same extractor to construct the model with the exact width before loading weights.

Q14. What are typical performance characteristics and hardware expectations?
A14. On CPU, training can take several minutes depending on dataset size; GPU/MPS significantly accelerates both training and inference. Sequence length and distance/velocity features increase dimensionality and memory use; batch size may need reduction on low-memory hardware. Real-time inference processes at webcam frame rate if the extractor and model fit in available compute.

Q15. How do you troubleshoot common issues?
A15. No data found: verify dataset/ contains .npy files per sign (run collect_data.py). Model load errors: ensure models/sign_model.pth exists and matches the current vocabulary size. Poor accuracy: collect more diverse samples, increase epochs, or simplify the vocabulary. Webcam/landmark failures: check lighting, camera access, and that MediaPipe task files exist in mediapipe/.

Q16. How is text-to-speech integrated and throttled?
A16. In [realtime/realtime_inference.py](realtime/realtime_inference.py), pyttsx3 is initialized once and invoked when a decoded sign differs from the last spoken phrase and a cooldown (default 1s) has elapsed. This prevents rapid repetition while keeping latency low.

Q17. What is stored after training and how is it reused?
A17. The best-performing state_dict is saved to models/sign_model.pth. Inference constructs the same GRU architecture, loads this state, and uses the vocabulary from [sign_vocab.py](sign_vocab.py) to map predicted IDs back to sign strings.

Q18. How could the system be extended academically or in production?
A18. Potential extensions include: replacing greedy decoding with a beam search CTC decoder for robustness, adding temporal augmentations during training, integrating confidence thresholds to suppress low-probability outputs, exporting to ONNX for mobile deployment, and adding evaluation scripts that compute per-class precision/recall on a held-out set.
