# handsign.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import mediapipe as mp
from collections import deque

class ASLRecognizer:
    def __init__(self, model_path='asl_model.h5', label_encoder_path='label_encoder.pkl'):
        # Load model and label encoder
        self.model = keras.models.load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.prediction_buffer = deque(maxlen=10)
        
        print("✅ ASL Recognizer initialized!")
        print(f"🎯 Loaded model with {len(self.label_encoder.classes_)} classes")
    
    def preprocess_hand_roi(self, frame, hand_landmarks):
        """Extract hand region from landmarks"""
        h, w = frame.shape[:2]
        
        # Get bounding box
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Extract and preprocess hand ROI
        hand_roi = frame[y_min:y_max, x_min:x_max]
        
        if hand_roi.size == 0:
            return None, (x_min, y_min, x_max, y_max)
        
        hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
        hand_roi = cv2.resize(hand_roi, (64, 64))
        hand_roi = hand_roi.astype('float32') / 255.0
        hand_roi = np.expand_dims(hand_roi, axis=0)
        
        return hand_roi, (x_min, y_min, x_max, y_max)
    
    def predict(self, hand_roi):
        """Predict gesture from hand ROI"""
        predictions = self.model.predict(hand_roi, verbose=0)
        confidence = np.max(predictions)
        predicted_idx = np.argmax(predictions)
        predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        return predicted_class, confidence
    
    def run_realtime(self):
        """Run real-time recognition"""
        cap = cv2.VideoCapture(0)
        
        print("🎥 Starting real-time recognition...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            current_prediction = "Show hand"
            current_confidence = 0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Predict gesture
                    hand_roi, bbox = self.preprocess_hand_roi(frame, hand_landmarks)
                    if hand_roi is not None:
                        prediction, confidence = self.predict(hand_roi)
                        
                        if confidence > 0.7:
                            self.prediction_buffer.append(prediction)
                        
                        # Get most common prediction from buffer
                        if len(self.prediction_buffer) > 0:
                            most_common = max(set(self.prediction_buffer), 
                                            key=self.prediction_buffer.count)
                            current_prediction = most_common
                            current_confidence = confidence
                        
                        # Draw bounding box and prediction
                        x_min, y_min, x_max, y_max = bbox
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, f'{prediction} ({confidence:.2f})', 
                                  (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display current prediction
            cv2.putText(frame, f'ASL: {current_prediction}', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'Confidence: {current_confidence:.2f}', (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('ASL Real-time Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = ASLRecognizer()
    recognizer.run_realtime()