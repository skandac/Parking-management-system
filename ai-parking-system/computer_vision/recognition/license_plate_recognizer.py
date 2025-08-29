"""
License plate recognition using OCR and computer vision techniques
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LicensePlateRecognizer:
    """
    AI-powered license plate recognition system
    """
    
    def __init__(self, tesseract_path: str = None):
        self.tesseract_path = tesseract_path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # License plate patterns for different countries
        self.plate_patterns = {
            'US': r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{1,2}$',
            'UK': r'^[A-Z]{2}[0-9]{2}\s?[A-Z]{3}$',
            'EU': r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{1,2}$',
            'IN': r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
        }
        
        # Preprocessing parameters
        self.kernel_size = (5, 5)
        self.blur_kernel = (3, 3)
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better license plate detection
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
            morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Invert image for better text detection
            thresh = cv2.bitwise_not(thresh)
            
            return thresh
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return image
    
    def detect_license_plate_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect potential license plate regions in the image
        """
        try:
            # Preprocess image
            processed = self.preprocess_image(image)
            
            # Find contours
            contours, _ = cv2.findContours(
                processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            plate_regions = []
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (license plates are typically rectangular)
                aspect_ratio = w / float(h)
                if 2.0 <= aspect_ratio <= 5.5:
                    # Filter by size
                    if w > 100 and h > 20:
                        # Extract region
                        region = image[y:y+h, x:x+w]
                        plate_regions.append({
                            'region': region,
                            'bbox': (x, y, w, h),
                            'confidence': 0.0
                        })
            
            return plate_regions
            
        except Exception as e:
            logger.error(f"Error in license plate region detection: {e}")
            return []
    
    def recognize_text(self, image: np.ndarray) -> str:
        """
        Recognize text from license plate image using OCR
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Configure OCR parameters
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Perform OCR
            text = pytesseract.image_to_string(pil_image, config=custom_config)
            
            # Clean and format text
            text = text.strip().replace('\n', '').replace(' ', '')
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            return text
            
        except Exception as e:
            logger.error(f"Error in text recognition: {e}")
            return ""
    
    def validate_license_plate(self, text: str, country: str = 'US') -> bool:
        """
        Validate license plate format based on country
        """
        if not text:
            return False
        
        pattern = self.plate_patterns.get(country, self.plate_patterns['US'])
        return bool(re.match(pattern, text))
    
    def process_image(self, image: np.ndarray, country: str = 'US') -> List[Dict[str, Any]]:
        """
        Complete license plate recognition pipeline
        """
        results = []
        
        try:
            # Detect license plate regions
            plate_regions = self.detect_license_plate_regions(image)
            
            for region_data in plate_regions:
                region = region_data['region']
                bbox = region_data['bbox']
                
                # Recognize text
                text = self.recognize_text(region)
                
                if text:
                    # Validate format
                    is_valid = self.validate_license_plate(text, country)
                    
                    # Calculate confidence based on text length and validation
                    confidence = min(len(text) / 8.0, 1.0) if is_valid else 0.3
                    
                    result = {
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence,
                        'is_valid': is_valid,
                        'country': country,
                        'region_image': region
                    }
                    
                    results.append(result)
            
            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in license plate processing: {e}")
        
        return results
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for better OCR results
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced = cv2.merge([l, a, b])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Increase contrast
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=10)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in image enhancement: {e}")
            return image
    
    def get_recognition_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics of license plate recognition
        """
        if not results:
            return {
                'total_plates': 0,
                'valid_plates': 0,
                'avg_confidence': 0.0,
                'countries': {}
            }
        
        valid_count = sum(1 for r in results if r['is_valid'])
        total_confidence = sum(r['confidence'] for r in results)
        
        countries = {}
        for result in results:
            country = result['country']
            if country not in countries:
                countries[country] = 0
            countries[country] += 1
        
        return {
            'total_plates': len(results),
            'valid_plates': valid_count,
            'avg_confidence': total_confidence / len(results),
            'countries': countries
        }
