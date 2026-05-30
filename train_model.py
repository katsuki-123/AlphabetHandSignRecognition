# train_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

def load_training_data():
    """Load training data from the subdirectories"""
    train_dir = 'ASL_Alphabet_Dataset/asl_alphabet_train'
    
    print("📥 Loading training data...")
    
    images = []
    labels = []
    
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"   Loading {class_name}...", end=" ")
            count = 0
            
            # Load only a subset to avoid memory issues (you can increase this)
            for img_file in list(os.listdir(class_dir))[:500]:  # First 500 images per class
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (64, 64))
                        img = img.astype('float32') / 255.0
                        images.append(img)
                        labels.append(class_name)
                        count += 1
            print(f"{count} images")
    
    return np.array(images), np.array(labels)

def load_test_data():
    """Load test data from individual files"""
    test_dir = 'ASL_Alphabet_Dataset/asl_alphabet_test'
    
    print("📥 Loading test data...")
    
    images = []
    labels = []
    
    for img_file in os.listdir(test_dir):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            # Extract class name from filename (A_test.jpg -> A)
            class_name = img_file.split('_')[0]
            
            img_path = os.path.join(test_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(class_name)
    
    print(f"   Loaded {len(images)} test images")
    return np.array(images), np.array(labels)

def create_model(num_classes):
    """Create CNN model"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    print("🚀 Starting ASL Model Training")
    print("=" * 50)
    
    try:
        # Load training data
        X_train, y_train = load_training_data()
        
        # Load test data
        X_test, y_test = load_test_data()
        
        if len(X_train) == 0 or len(X_test) == 0:
            print("❌ No images loaded! Check your dataset.")
            return
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Combine all labels to create consistent encoding
        all_labels = np.concatenate([y_train, y_test])
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)
        
        # Encode both sets
        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        print(f"🎯 Classes: {list(label_encoder.classes_)}")
        print(f"📈 Number of classes: {len(label_encoder.classes_)}")
        
        # Create and compile model
        model = create_model(len(label_encoder.classes_))
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n🔄 Starting training...")
        history = model.fit(
            X_train, y_train_encoded,
            epochs=15,
            batch_size=32,
            validation_data=(X_test, y_test_encoded),
            verbose=1
        )
        
        # Save model and label encoder
        model.save('asl_model.h5')
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print("✅ Training completed!")
        print("💾 Model saved as 'asl_model.h5'")
        print("💾 Label encoder saved as 'label_encoder.pkl'")
        
        # Evaluate final model
        test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=0)
        print(f"🎯 Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()