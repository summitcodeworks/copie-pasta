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
    """Advanced mobile display panel detector - FIXED for false positives"""
    
    def __init__(self):
        # INCREASED confidence threshold to reduce false positives
        self.min_confidence = 0.70  # Increased from 0.4 to 0.7
        # Only detect actual cell phones
        self.phone_class = 'cell phone'
        
    def detect_with_yolo(self, image):
        """Detect using YOLOv8 - STRICT mode to prevent false positives"""
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
                
                # STRICT: Only accept 'cell phone' class with high confidence
                if confidence > self.min_confidence and class_name == self.phone_class:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # Validate aspect ratio - typical for mobile phones
                    # Portrait: 0.45-0.75, Landscape: 1.33-2.2
                    if (0.45 <= aspect_ratio <= 0.75) or (1.33 <= aspect_ratio <= 2.2):
                        # Additional validation: check if size is reasonable
                        img_height, img_width = image.shape[:2]
                        area_ratio = (w * h) / (img_width * img_height)
                        
                        # Panel should be between 8% and 80% of image
                        if 0.08 <= area_ratio <= 0.80:
                            detections.append({
                                'bbox': (x1, y1, w, h),
                                'confidence': confidence,
                                'method': 'YOLO',
                                'class': class_name,
                                'aspect_ratio': aspect_ratio,
                                'area_ratio': area_ratio
                            })
        
        return detections
    
    def detect(self, image):
        """Main detection method - YOLO only for maximum accuracy"""
        # ONLY use YOLO to prevent false positives from tables/lights
        # Contour and color methods are disabled as they cause false positives
        yolo_detections = self.detect_with_yolo(image)
        return yolo_detections


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
        font = cv2.FONT_HERSHEY_SIMPLEX
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
            font = cv2.FONT_HERSHEY_SIMPLEX
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
        (text_width, text_height), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = (width - text_width) // 2
        text_y = 50
        
        # Warning icon (X)
        icon_size = 35
        icon_x = text_x - icon_size - 20
        icon_y = text_y - icon_size//2
        cv2.line(marked, (icon_x, icon_y), (icon_x+icon_size, icon_y+icon_size), (255, 255, 255), 5)
        cv2.line(marked, (icon_x+icon_size, icon_y), (icon_x, icon_y+icon_size), (255, 255, 255), 5)
        
        cv2.putText(marked, warning_text, (text_x+2, text_y+2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(marked, warning_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
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
        "version": "2.1 - False Positive Fix",
        "confidence_threshold": detector.min_confidence
    })


@app.route('/detect', methods=['POST'])
def detect_panel():
    """
    Main endpoint to detect mobile panel in uploaded image
    Uses strict YOLOv8 detection to prevent false positives
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
        
        # Detect panels using strict YOLO-only method
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
                    "class": d.get('class', 'Unknown'),
                    "aspect_ratio": round(d.get('aspect_ratio', 0), 2),
                    "area": d['bbox'][2] * d['bbox'][3]
                }
                for idx, d in enumerate(detections)
            ],
            "marked_image": f"data:image/jpeg;base64,{marked_image_base64}",
            "message": f"Detected {len(detections)} mobile panel(s)" if detections else "No mobile panels detected",
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "detection_mode": "strict_yolo_only"
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
                    avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0.0
                    
                    results.append({
                        "filename": file.filename,
                        "detected": len(detections) > 0,
                        "panel_count": len(detections),
                        "confidence": round(avg_confidence, 3),
                        "result": "NG" if detections else "OK"
                    })
        
        return jsonify({
            "total_images": len(results),
            "ng_count": sum(1 for r in results if r['detected']),
            "ok_count": sum(1 for r in results if not r['detected']),
            "results": results,
            "detection_mode": "strict_yolo_only"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/config', methods=['GET'])
def get_config():
    """Get current detection configuration"""
    return jsonify({
        "min_confidence": detector.min_confidence,
        "detection_class": detector.phone_class,
        "yolo_model": "yolov8n.pt",
        "detection_methods": ["YOLO-only (strict)"],
        "false_positive_prevention": "enabled",
        "aspect_ratio_validation": "enabled",
        "area_ratio_validation": "enabled"
    })


@app.route('/config', methods=['POST'])
def update_config():
    """Update detection configuration"""
    try:
        data = request.json
        if 'min_confidence' in data:
            new_confidence = float(data['min_confidence'])
            if 0.0 <= new_confidence <= 1.0:
                detector.min_confidence = new_confidence
                return jsonify({
                    "success": True,
                    "message": f"Confidence threshold updated to {new_confidence}",
                    "new_config": {
                        "min_confidence": detector.min_confidence
                    }
                })
            else:
                return jsonify({"error": "Confidence must be between 0.0 and 1.0"}), 400
        
        return jsonify({"error": "No valid parameters provided"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Advanced Mobile Panel Detection API v2.1")
    print("=" * 60)
    print("✓ YOLO Model:", "Loaded" if model else "Not available")
    print("✓ Detection Mode: STRICT (YOLO-only)")
    print("✓ False Positive Prevention: ENABLED")
    print("✓ Confidence Threshold:", detector.min_confidence)
    print("✓ Features: Aspect ratio + Area validation")
    print("\n📡 Endpoints:")
    print("  - POST /detect          (JSON with base64 image)")
    print("  - POST /detect/image    (Direct image file)")
    print("  - POST /detect/batch    (Multiple images)")
    print("  - GET  /health          (Health check)")
    print("  - GET  /config          (View settings)")
    print("  - POST /config          (Update settings)")
    print("\n🌐 Server starting on http://0.0.0.0:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
