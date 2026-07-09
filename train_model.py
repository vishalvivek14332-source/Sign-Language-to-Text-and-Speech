import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# Configure TensorFlow to use memory efficiently
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Configuration
DATASET_PATH = 'AtoZ_3.1'
IMG_HEIGHT = 200  # Reduced from 400 to 200
IMG_WIDTH = 200   # Reduced from 400 to 200
BATCH_SIZE = 16   # Reduced from 32 to 16
EPOCHS = 15       # Increased slightly since images are smaller
NUM_CLASSES = 26  # A-Z
VALIDATION_SPLIT = 0.2
MODEL_NAME = 'cnn8grps_rad1_model.h5'

print("=" * 60)
print("SIGN LANGUAGE RECOGNITION - MODEL TRAINING (OPTIMIZED)")
print("=" * 60)
print(f"Dataset path: {DATASET_PATH}")
print(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH} (reduced for memory efficiency)")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Classes: {NUM_CLASSES} (A-Z)")
print(f"Validation split: {VALIDATION_SPLIT*100}%")
print("=" * 60)

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset folder '{DATASET_PATH}' not found!")
    print("Please make sure the AtoZ_3.1 folder exists in the current directory.")
    exit(1)

# Count images per class
print("\nChecking dataset structure...")
class_counts = {}
for class_name in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_name] = count
        print(f"  Class {class_name}: {count} images")

total_images = sum(class_counts.values())
print(f"\nTotal images: {total_images}")
print(f"Expected per class: ~{total_images//NUM_CLASSES} images")

# 1. Create ImageDataGenerator with validation split and data augmentation
print("\n[1/5] Creating data generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=VALIDATION_SPLIT
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

# 2. Load training data
print("[2/5] Loading training data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# 3. Load validation data
print("[3/5] Loading validation data...")
validation_generator = validation_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Class mapping: {train_generator.class_indices}")

# 4. Build Optimized CNN Model (much smaller than before)
print("\n[4/5] Building optimized CNN model...")

model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    
    # Fourth Convolutional Block (smaller filters)
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    
    # Global Average Pooling instead of Flatten (dramatically reduces parameters)
    layers.GlobalAveragePooling2D(),
    
    # Dense Layers (much smaller)
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    # Output Layer (26 classes)
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Display model summary
model.summary()

# Calculate total parameters
total_params = model.count_params()
print(f"\nTotal trainable parameters: {total_params:,}")
print(f"Memory estimate: ~{total_params * 4 / (1024**2):.2f} MB (for parameters only)")

# 5. Compile the model
print("\n[5/5] Compiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Callbacks for better training
callbacks = [
    # Save the best model based on validation accuracy
    keras.callbacks.ModelCheckpoint(
        'best_' + MODEL_NAME,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Early stopping to prevent overfitting
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when plateau is reached
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

print("\n" + "=" * 60)
print("STARTING TRAINING...")
print("=" * 60)

# Calculate steps
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# 7. Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# 8. Save the final model
print("\n" + "=" * 60)
print(f"Saving model as {MODEL_NAME}...")
model.save(MODEL_NAME)
print(f"Model saved successfully!")

# Also save in TensorFlow SavedModel format for compatibility
model.save('saved_model')
print(f"Backup model saved in 'saved_model' folder")

# 9. Plot training history
print("\nGenerating training plots...")

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot training & validation accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot training & validation loss
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

print("\nTraining plot saved as 'training_history.png'")

# 10. Final evaluation
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"Validation Loss: {val_loss:.4f}")

# Evaluate on training set (use a subset for speed)
train_loss, train_accuracy = model.evaluate(train_generator, verbose=0, steps=50)
print(f"Training Accuracy (sample): {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Training Loss (sample): {train_loss:.4f}")

# 11. Test with a sample prediction
print("\n" + "=" * 60)
print("SAMPLE PREDICTION TEST")
print("=" * 60)

# Get one batch from validation generator
x_batch, y_batch = next(validation_generator)
predictions = model.predict(x_batch[:5])

# Show predictions for first 5 images
class_indices_rev = {v: k for k, v in train_generator.class_indices.items()}
for i in range(5):
    true_class = class_indices_rev[np.argmax(y_batch[i])]
    pred_class = class_indices_rev[np.argmax(predictions[i])]
    confidence = np.max(predictions[i]) * 100
    print(f"Image {i+1}: True={true_class}, Predicted={pred_class} ({confidence:.1f}% confidence)")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nNext steps:")
print(f"1. Use '{MODEL_NAME}' in your main application (final_pred.py)")
print(f"2. Run: python final_pred.py")
print("\nHappy Sign Language Recognition! 🤟")