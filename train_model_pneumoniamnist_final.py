import tensorflow as tf
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# Suppress CUDA warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
print("Available GPUs:", physical_devices)

# Clear GPU memory
tf.keras.backend.clear_session()
gc.collect()

# Enable mixed precision
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Load PneumoniaMNIST dataset
data_dir = '/boot/XRAY DATA'
try:
    train_images = np.load(os.path.join(data_dir, 'train_images.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    val_images = np.load(os.path.join(data_dir, 'val_images.npy'))
    val_labels = np.load(os.path.join(data_dir, 'val_labels.npy'))
    try:
        test_images = np.load(os.path.join(data_dir, 'test_images.npy'))
        test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
        print("Test images shape:", test_images.shape)
        print("Test labels shape:", test_labels.shape)
    except FileNotFoundError:
        print("Test set not found. Skipping test evaluation.")
        test_images, test_labels = None, None
except FileNotFoundError:
    print("Dataset files not found in /boot/XRAY DATA. Ensure correct path and files.")
    raise

# Verify and fix train_images shape
expected_train_size = 3882 * 28 * 28
if train_images.size != expected_train_size:
    print(f"Error: train_images has size {train_images.size}, expected {expected_train_size}. Re-download dataset.")
    raise ValueError("Dataset corrupted; incorrect image size.")
if train_images.shape[0] != train_labels.shape[0]:
    print(f"Error: Mismatch between train_images ({train_images.shape[0]}) and train_labels ({train_labels.shape[0]}).")
    raise ValueError("Dataset corrupted; image and label counts do not match.")
train_images = train_images.reshape(3882, 28, 28)

# Oversample Normal class (3x)
normal_indices = np.where(train_labels.flatten() == 0)[0]
normal_images = train_images[normal_indices]
normal_labels = train_labels[normal_indices]
oversampled_normal_images = np.repeat(normal_images, 3, axis=0)  # ~1,152 Normal images
oversampled_normal_labels = np.repeat(normal_labels, 3, axis=0)
train_images = np.concatenate([train_images, oversampled_normal_images], axis=0)
train_labels = np.concatenate([train_labels, oversampled_normal_labels], axis=0)
print("Oversampled train images shape:", train_images.shape)
print("Oversampled train labels shape:", train_labels.shape)

# Reshape images
train_images = train_images.reshape(-1, 28, 28, 1)
val_images = val_images.reshape(-1, 28, 28, 1)
if test_images is not None:
    test_images = test_images.reshape(-1, 28, 28, 1)

# Preprocess images
def preprocess_image(image, label):
    image = tf.ensure_shape(image, [28, 28, 1])
    image = tf.image.resize(image, [299, 299])
    image = tf.image.grayscale_to_rgb(image)
    image = image / 255.0
    return image, label

# Enhanced augmentation
@tf.function
def augment_image(image, label):
    image = tf.ensure_shape(image, [299, 299, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    # Zoom
    scale = tf.random.uniform([], 0.9, 1.1)
    image = tf.image.resize(image, [int(299 * scale), int(299 * scale)])
    image = tf.image.resize_with_crop_or_pad(image, 299, 299)
    # Translation
    image = tf.image.random_crop(image, [299, 299, 3])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# Create datasets
batch_size = 8
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
if test_images is not None:
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Apply preprocessing and augmentation
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Verify dataset output
for x, y in train_dataset.take(1):
    print("Processed train batch shape:", x.shape, "Label shape:", y.shape)
for x, y in val_dataset.take(1):
    print("Processed validation batch shape:", x.shape, "Label shape:", y.shape)
if test_images is not None:
    for x, y in test_dataset.take(1):
        print("Processed test batch shape:", x.shape, "Label shape:", y.shape)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels.flatten())
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Define Inception-V3 model with extra dense layer
base_model = tf.keras.applications.InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool'),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), dtype='float32', name='dense_1'),
    tf.keras.layers.BatchNormalization(name='batch_norm_1'),
    tf.keras.layers.Dropout(0.5, name='dropout_1'),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), dtype='float32', name='dense_2'),
    tf.keras.layers.BatchNormalization(name='batch_norm_2'),
    tf.keras.layers.Dropout(0.3, name='dropout_2'),
    tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32', name='output')
])

# Build model
model.build((None, 299, 299, 3))

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('/content/model_initial.keras', monitor='val_loss', save_best_only=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ],
    verbose=1
)

# Fine-tune model
print("\nFine-tuning model...")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-7),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history_finetune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('/content/model_finetuned.keras', monitor='val_loss', save_best_only=True, mode='min')
    ],
    verbose=1
)

# Save final model
model.save('/content/model_final.keras')

# Evaluate on validation set
print("\nEvaluating model on validation set...")
val_loss, val_accuracy = model.evaluate(val_dataset, verbose=1)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on test set with threshold tuning
if test_images is not None:
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Metrics with threshold 0.15
    test_predictions = model.predict(test_dataset, verbose=1)
    test_pred_probs = test_predictions.flatten()
    test_pred_labels = (test_predictions > 0.15).astype(int).flatten()
    
    print("\nTest Set Metrics (Threshold=0.15):")
    auc_score = roc_auc_score(test_labels, test_pred_probs)
    print(f"AUC: {auc_score:.4f}")
    print("Classification Report:")
    print(classification_report(test_labels, test_pred_labels, target_names=['Normal', 'Pneumonia'], zero_division=0))
    print("Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_pred_labels)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix (Threshold=0.15)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
    plt.close()

# Training progress plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_finetune.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_finetune.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'] + history_finetune.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# ROC Curve
if test_images is not None:
    fpr, tpr, _ = roc_curve(test_labels, test_pred_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

# Before and after augmentation visualization (original size)
if test_images is not None:
    print("\nVisualizing original and augmented images...")
    for idx in [0, 1]:
        original_image = test_images[idx]
        test_label = test_labels[idx]
        
        test_image_tensor = tf.convert_to_tensor(original_image, dtype=tf.float32)
        test_label_tensor = tf.convert_to_tensor(test_label, dtype=tf.int32)
        
        preprocessed_image, _ = preprocess_image(test_image_tensor, test_label_tensor)
        augmented_image, _ = augment_image(preprocessed_image, test_label_tensor)
        
        prediction = model.predict(tf.expand_dims(preprocessed_image, 0), verbose=0)
        pred_label = "Pneumonia" if prediction[0] > 0.15 else "Normal"
        true_label = "Pneumonia" if test_label == 1 else "Normal"
        
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image.squeeze(), cmap='gray')
        plt.title(f"Original (28x28)\nTrue: {true_label}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(augmented_image.numpy()[:, :, 0], cmap='gray')
        plt.title(f"Augmented (299x299)\nPredicted: {pred_label}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.close()

# Verify saved model
print("\nVerifying saved model...")
class Cast(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, self.compute_dtype)

with tf.keras.utils.custom_object_scope({'Cast': Cast}):
    loaded_model = tf.keras.models.load_model('/content/model_finetuned.keras')
    val_loss_loaded, val_accuracy_loaded = loaded_model.evaluate(val_dataset, verbose=1)
    print(f"Loaded Model Validation Loss: {val_loss_loaded:.4f}, Validation Accuracy: {val_accuracy_loaded:.4f}")

# Clear GPU memory
tf.keras.backend.clear_session()
gc.collect()

# Check GPU memory
print("\nChecking GPU memory usage...")
!nvidia-smi
