import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load Dataset and Split (80% train, 20% validation)
# -----------------------------
dataset_path = "hand_sign_dataset"  # Folder containing class subfolders

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # 80% training, 20% validation
)

# Training data
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation data
val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False
) 

print("\n Data loaded successfully!")
print("Classes found:", train_data.class_indices)
print(f"Training samples: {train_data.samples}")
print(f"Validation samples: {val_data.samples}")

# -----------------------------
# 2. Build CNN Model
# -----------------------------
model = Sequential([
    tf.keras.Input(shape=(128, 128, 3)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 3. Train Model
# -----------------------------
epochs = 10
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    verbose=1
)

# -----------------------------
# 4. Evaluate Model
# -----------------------------
loss, acc = model.evaluate(val_data, verbose=0)
print(f"\n Validation Accuracy: {acc*100:.2f}%")
print(f" Validation Loss: {loss:.4f}")

# -----------------------------
# 5. Plot Accuracy & Loss Graphs
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# 6. Save Model
# -----------------------------
os.makedirs("saved_model", exist_ok=True)
save_path = "saved_model/hand_sign_cnn_model.h5"
model.save(save_path)

print(f"\n Model saved successfully to '{save_path}'")
