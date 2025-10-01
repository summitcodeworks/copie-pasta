from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import os
from ultralytics import YOLO
import torch

app = Flask(__name__)
CORS(app)

# Initialize YOLO model (will download automatically on first run)
try:
    # Using YOLOv8 for better detection
    model = YOLO('yolov8n.pt')  # Nano model for speed, use 'yolov8x.pt' for accuracy
    print("✓ YOLO model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    model = None


class AdvancedPanelDetector:
    """Advanced mobile display panel detector using multiple methods"""
    
    def __init__(self):
        self.min_confidence = 0.4
        self.phone_classes = ['cell phone', 'phone', 'mobile phone']
        
    def preprocess_image(self, image):
        """Enhanced preprocessing for better detection"""
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Enhance contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_with_yolo(self, image):
        """Detect using YOLOv8 deep learning model"""
        if model is None:
            return []
        
        results = model(image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Check if it's a phone/mobile device
                if confidence > self.min_confidence and class_name in self.phone_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'confidence': confidence,
                        'method': 'YOLO',
                        'class': class_name
                    })
        
        return detections
    
    def detect_with_contours(self, image):
        """Advanced contour-based detection"""
        preprocessed = self.preprocess_image(image)
        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        
        # Multiple edge detection approaches
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        height, width = image.shape[:2]
        min_area = (height * width) * 0.05
        max_area = (height * width) * 0.95
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Mobile panels have specific aspect ratios
                # Portrait: 0.4-0.7, Landscape: 1.4-2.5
                if (0.4 <= aspect_ratio <= 0.75) or (1.4 <= aspect_ratio <= 2.5):
                    # Calculate additional features
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Approximate polygon
                    epsilon = 0.02 * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Rectangularity check (should be close to 4 corners)
                    if len(approx) >= 4 and len(approx) <= 8:
                        confidence = min(0.9, (area / max_area) * 0.7 + circularity * 0.3)
                        
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'method': 'Contour',
                            'aspect_ratio': aspect_ratio
                        })
        
        return detections
    
    def detect_with_color(self, image):
        """Detect based on typical mobile panel colors (screens)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for typical screens (blue-ish glow)
        # Range 1: Blue screens
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Range 2: White/bright screens
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
        mask2 = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        height, width = image.shape[:2]
        min_area = (height * width) * 0.03
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                if (0.4 <= aspect_ratio <= 0.75) or (1.4 <= aspect_ratio <= 2.5):
                    confidence = min(0.85, area / (height * width * 0.5))
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'method': 'Color'
                    })
        
        return detections
    
    def merge_detections(self, detections):
        """Merge overlapping detections using Non-Maximum Suppression"""
        if not detections:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Convert to x1, y1, x2, y2 format
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = boxes[:, 2] * boxes[:, 3]
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= 0.3)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def detect(self, image):
        """Main detection method combining all approaches"""
        all_detections = []
        
        # Method 1: YOLO (most accurate)
        yolo_detections = self.detect_with_yolo(image)
        all_detections.extend(yolo_detections)
        
        # Method 2: Advanced contour detection
        contour_detections = self.detect_with_contours(image)
        all_detections.extend(contour_detections)
        
        # Method 3: Color-based detection
        color_detections = self.detect_with_color(image)
        all_detections.extend(color_detections)
        
        # Merge overlapping detections
        final_detections = self.merge_detections(all_detections)
        
        return final_detections


def mark_image(image, detections):
    """Mark detected panels on the image with professional annotations"""
    marked = image.copy()
    height, width = marked.shape[:2]
    
    if not detections:
        # No panel detected - show OK
        # Create semi-transparent overlay
        overlay = marked.copy()
        
        # Add green banner at top
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 180, 0), -1)
        marked = cv2.addWeighted(overlay, 0.3, marked, 0.7, 0)
        
        # Add OK text
        font = cv2.FONT_HERSHEY_BOLD
        text = "OK - No Panel Detected"
        font_scale = 1.2
        thickness = 3
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (width - text_width) // 2
        y = 50
        
        # Text shadow
        cv2.putText(marked, text, (x+2, y+2), font, font_scale, (0, 0, 0), thickness+1)
        # Main text
        cv2.putText(marked, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # Add checkmark icon
        check_size = 40
        check_x = x - check_size - 20
        check_y = y - check_size//2
        cv2.line(marked, (check_x, check_y), (check_x+check_size//3, check_y+check_size//2), (255, 255, 255), 4)
        cv2.line(marked, (check_x+check_size//3, check_y+check_size//2), (check_x+check_size, check_y-check_size//2), (255, 255, 255), 4)
        
    else:
        # Panel detected - show NG
        for idx, detection in enumerate(detections, 1):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            method = detection.get('method', 'Unknown')
            
            # Draw thick red rectangle
            cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 4)
            
            # Draw corners for emphasis
            corner_length = 30
            corner_thickness = 6
            # Top-left
            cv2.line(marked, (x, y), (x + corner_length, y), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x, y), (x, y + corner_length), (0, 0, 255), corner_thickness)
            # Top-right
            cv2.line(marked, (x + w, y), (x + w - corner_length, y), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x + w, y), (x + w, y + corner_length), (0, 0, 255), corner_thickness)
            # Bottom-left
            cv2.line(marked, (x, y + h), (x + corner_length, y + h), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x, y + h), (x, y + h - corner_length), (0, 0, 255), corner_thickness)
            # Bottom-right
            cv2.line(marked, (x + w, y + h), (x + w - corner_length, y + h), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x + w, y + h), (x + w, y + h - corner_length), (0, 0, 255), corner_thickness)
            
            # NG Label with background
            font = cv2.FONT_HERSHEY_BOLD
            label = "NG"
            font_scale = 2.5
            thickness = 4
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position above the box
            label_y = max(y - 10, text_height + 20)
            label_x = x
            
            # Draw red background with some padding
            padding = 15
            cv2.rectangle(marked, 
                         (label_x - padding, label_y - text_height - padding),
                         (label_x + text_width + padding, label_y + baseline + padding),
                         (0, 0, 255), -1)
            
            # Draw white border
            cv2.rectangle(marked, 
                         (label_x - padding, label_y - text_height - padding),
                         (label_x + text_width + padding, label_y + baseline + padding),
                         (255, 255, 255), 2)
            
            # Draw text shadow
            cv2.putText(marked, label, (label_x + 2, label_y + 2),
                       font, font_scale, (0, 0, 0), thickness + 1)
            # Draw main text
            cv2.putText(marked, label, (label_x, label_y),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Add confidence and method info
            info_text = f"#{idx} {confidence*100:.1f}% ({method})"
            info_font_scale = 0.6
            info_thickness = 2
            
            (info_width, info_height), _ = cv2.getTextSize(info_text, font, info_font_scale, info_thickness)
            info_y = y + h + 25
            
            # Info background
            cv2.rectangle(marked,
                         (x, info_y - info_height - 5),
                         (x + info_width + 10, info_y + 5),
                         (0, 0, 0), -1)
            cv2.rectangle(marked,
                         (x, info_y - info_height - 5),
                         (x + info_width + 10, info_y + 5),
                         (0, 0, 255), 2)
            
            cv2.putText(marked, info_text, (x + 5, info_y),
                       font, info_font_scale, (255, 255, 255), info_thickness)
        
        # Add warning banner at top
        overlay = marked.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
        marked = cv2.addWeighted(overlay, 0.3, marked, 0.7, 0)
        
        warning_text = f"PANEL DETECTED - NG ({len(detections)} found)"
        font_scale = 1.0
        thickness = 3
        (text_width, text_height), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_BOLD, font_scale, thickness)
        text_x = (width - text_width) // 2
        text_y = 50
        
        # Warning icon (X)
        icon_size = 35
        icon_x = text_x - icon_size - 20
        icon_y = text_y - icon_size//2
        cv2.line(marked, (icon_x, icon_y), (icon_x+icon_size, icon_y+icon_size), (255, 255, 255), 5)
        cv2.line(marked, (icon_x+icon_size, icon_y), (icon_x, icon_y+icon_size), (255, 255, 255), 5)
        
        cv2.putText(marked, warning_text, (text_x+2, text_y+2),
                   cv2.FONT_HERSHEY_BOLD, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(marked, warning_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_BOLD, font_scale, (255, 255, 255), thickness)
    
    return marked


# Initialize detector
detector = AdvancedPanelDetector()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Advanced Mobile Panel Detection API",
        "yolo_loaded": model is not None,
        "version": "2.0"
    })


@app.route('/detect', methods=['POST'])
def detect_panel():
    """
    Main endpoint to detect mobile panel in uploaded image
    Uses advanced deep learning and computer vision techniques
    """
    try:
        # Read image from request
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        elif request.is_json and 'image' in request.json:
            base64_image = request.json['image']
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            
            image_bytes = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": "No image provided"}), 400
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Detect panels using advanced methods
        detections = detector.detect(image)
        
        # Mark the image
        marked_image = mark_image(image, detections)
        
        # Encode marked image
        _, buffer = cv2.imencode('.jpg', marked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        marked_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate overall confidence
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0.0
        
        # Prepare detailed response
        response = {
            "detected": len(detections) > 0,
            "result": "NG - Panel Detected" if detections else "OK - No Panel Detected",
            "panel_count": len(detections),
            "confidence": round(avg_confidence, 3),
            "detections": [
                {
                    "id": idx + 1,
                    "bbox": d['bbox'],
                    "confidence": round(d['confidence'], 3),
                    "method": d.get('method', 'Unknown'),
                    "area": d['bbox'][2] * d['bbox'][3]
                }
                for idx, d in enumerate(detections)
            ],
            "marked_image": f"data:image/jpeg;base64,{marked_image_base64}",
            "message": f"Detected {len(detections)} mobile panel(s)" if detections else "No mobile panels detected",
            "image_size": {"width": image.shape[1], "height": image.shape[0]}
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/detect/image', methods=['POST'])
def detect_panel_return_image():
    """Returns the marked image directly as a file"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Detect and mark
        detections = detector.detect(image)
        marked_image = mark_image(image, detections)
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', marked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        byte_io = BytesIO(buffer)
        byte_io.seek(0)
        
        status = "NG" if detections else "OK"
        return send_file(
            byte_io,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'marked_{status}_{len(detections)}panels.jpg'
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect/batch', methods=['POST'])
def detect_batch():
    """Process multiple images at once"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files provided"}), 400
        
        results = []
        for file in files:
            if file.filename:
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    detections = detector.detect(image)
                    
           
