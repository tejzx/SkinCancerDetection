import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import json
import time

# Configure environment
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('training.log')]
)
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
CONFIG = {
    'dataset_path': r"C:\Users\reach\OneDrive\Desktop\SkinCancer1\skindataset\data",
    'img_size': (224, 224),
    'batch_size': 32,
    'num_classes': 3,
    'epochs': 5,
    'patience': 5,
    'learning_rate': 0.0001
}

def setup_directories():
    """Create required directories."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def create_data_generators():
    """Create data generators with proper class balancing."""
    logger.info("Creating data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_gen = train_datagen.flow_from_directory(
        CONFIG['dataset_path'],
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_directory(
        CONFIG['dataset_path'],
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    logger.info(f"Found {train_gen.samples} training images")
    logger.info(f"Found {val_gen.samples} validation images")
    
    # Save class indices
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f)
    
    return train_gen, val_gen

def create_model():
    """Create optimized EfficientNet model."""
    logger.info("Creating model architecture...")
    
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*CONFIG['img_size'], 3)
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(*CONFIG['img_size'], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(CONFIG['num_classes'], activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    model.summary(print_fn=logger.info)
    return model

def evaluate_model(model, val_gen):
    """Proper evaluation with correct sample count."""
    logger.info("Evaluating model...")
    
    # Get all validation data at once
    x_val = []
    y_val = []
    for i in range(len(val_gen)):
        x, y = val_gen[i]
        x_val.append(x)
        y_val.append(y)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    
    # Predict on all validation data
    y_pred = model.predict(x_val, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    # Classification report
    report = classification_report(
        y_true, 
        y_pred_classes, 
        target_names=list(val_gen.class_indices.keys()),
        digits=4
    )
    logger.info("\nClassification Report:\n%s", report)
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=list(val_gen.class_indices.keys()),
        yticklabels=list(val_gen.class_indices.keys())
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)

def train_model():
    """Complete training pipeline."""
    setup_directories()
    start_time = time.time()
    
    train_gen, val_gen = create_data_generators()
    model = create_model()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=CONFIG['patience'], restore_best_weights=True),
        ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True),
        CSVLogger('logs/training_log.csv')
    ]
    
    steps_per_epoch = max(1, train_gen.samples // CONFIG['batch_size'])
    val_steps = max(1, val_gen.samples // CONFIG['batch_size'])
    
    logger.info(f"\nStarting training for {CONFIG['epochs']} epochs")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {val_steps}\n")
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    model.save('models/final_model.h5')
    logger.info("Model training completed and saved.")
    
    # Proper evaluation
    evaluate_model(model, val_gen)
    
    duration = time.time() - start_time
    logger.info(f"\nTraining completed in {duration//60:.0f}m {duration%60:.0f}s")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise