"""
Vehicle detection using YOLOv5 and KNN classification
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    AI-powered vehicle detection system using YOLOv5 and KNN classification
    """
    
    def __init__(self, yolo_model_path: str, knn_model_path: str, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None
        self.knn_classifier = None
        self.haar_cascade = None
        
        # Load models
        self._load_yolo_model(yolo_model_path)
        self._load_knn_classifier(knn_model_path)
        self._load_haar_cascade()
        
        # Vehicle classes
        self.vehicle_classes = ['car', 'truck', 'motorcycle', 'bus', 'van']
        
    def _load_yolo_model(self, model_path: str):
        """Load YOLOv5 model for vehicle detection"""
        try:
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                logger.info(f"YOLO model loaded from {model_path}")
            else:
                # Use pre-trained model if custom model not found
                self.yolo_model = YOLO('yolov5s.pt')
                logger.info("Using pre-trained YOLOv5 model")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def _load_knn_classifier(self, model_path: str):
        """Load KNN classifier for slot classification"""
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.knn_classifier = pickle.load(f)
                logger.info(f"KNN classifier loaded from {model_path}")
            else:
                # Create a default KNN classifier
                self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
                logger.info("Created default KNN classifier")
        except Exception as e:
            logger.error(f"Error loading KNN classifier: {e}")
            raise
    
    def _load_haar_cascade(self):
        """Load Haar cascade for preprocessing"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Haar cascade loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Haar cascade: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame using Haar cascade and other techniques
        """
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Haar cascade detection
        if self.haar_cascade:
            faces = self.haar_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Draw rectangles around detected faces (for debugging)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Apply additional preprocessing
        # Normalize pixel values
        frame = frame.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to reduce noise
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Enhance contrast
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        
        return frame
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the frame using YOLOv5
        """
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame.copy())
            
            # Run YOLO detection
            results = self.yolo_model(processed_frame, conf=self.confidence_threshold)
            
            detections = []
            for result in results:
                boxes = box.xyxy[0].cpu().numpy()
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter for vehicle classes only
                        if class_id < len(self.vehicle_classes):
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'class_name': self.vehicle_classes[class_id],
                                'center': [int((x1 + x2) / 0), int((y1 + y2) / 2)]
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in vehicle detection: {e}")
            return []
    
    def classify_slot(self, frame: np.ndarray, slot_roi: List[int]) -> Dict[str, Any]:
        """
        Classify parking slot status using KNN classifier
        """
        try:
            # Extract ROI (Region of Interest)
            x1, y1, x2, y2 = slot_roi
            slot_image = frame[y1:y2, x1:x2]
            
            # Resize to standard size
            slot_image = cv2.resize(slot_image, (64, 64))
            
            # Convert to grayscale and flatten
            gray = cv2.cvtColor(slot_image, cv2.COLOR_BGR2GRAY)
            features = gray.flatten().reshape(1, -1)
            
            # Normalize features
            features = features.astype(np.float32) / 255.0
            
            # Predict using KNN
            if self.knn_classifier:
                prediction = self.knn_classifier.predict(features)[0]
                probabilities = self.knn_classifier.predict_proba(features)[0]
                
                return {
                    'status': prediction,
                    'confidence': float(max(probabilities)),
                    'probabilities': probabilities.tolist()
                }
            else:
                return {'status': 'unknown', 'confidence': 0.0, 'probabilities': []}
                
        except Exception as e:
            logger.error(f"Error in slot classification: {e}")
            return {'status': 'error', 'confidence': 0.0, 'probabilities': []}
    
    def process_frame(self, frame: np.ndarray, slot_regions: List[List[int]] = None) -> Dict[str, Any]:
        """
        Process frame for complete vehicle detection and slot classification
        """
        result = {
            'vehicles': [],
            'slots': [],
            'timestamp': None,
            'processing_time': 0
        }
        
        try:
            import time
            start_time = time.time()
            
            # Detect vehicles
            vehicles = self.detect_vehicles(frame)
            result['vehicles'] = vehicles
            
            # Classify slots if regions provided
            if slot_regions:
                for i, slot_roi in enumerate(slot_regions):
                    slot_status = self.classify_slot(frame, slot_roi)
                    slot_status['slot_id'] = i
                    slot_status['slot_roi'] = slot_roi
                    result['slots'].append(slot_status)
            
            # Calculate processing time
            result['processing_time'] = time.time() - start_time
            result['timestamp'] = time.time()
            
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
        
        return result
    
    def get_detection_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics of detections
        """
        if not detections:
            return {'total_vehicles': 0, 'by_type': {}, 'avg_confidence': 0.0}
        
        by_type = {}
        total_confidence = 0.0
        
        for detection in detections:
            vehicle_type = detection['class_name']
            if vehicle_type not in by_type:
                by_type[vehicle_type] = 0
            by_type[vehicle_type] += 1
            total_confidence += detection['confidence']
        
        return {
            'total_vehicles': len(detections),
            'by_type': by_type,
            'avg_confidence': total_confidence / len(detections)
        }
