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
    model = YOLO('yolov8n.pt')
    print("✓ YOLO model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    model = None


class AdvancedPanelDetector:
    """Hybrid panel detector - Detects panels but filters out lights/tables"""
    
    def __init__(self):
        # Lower confidence for better detection
        self.min_yolo_confidence = 0.40
        self.min_contour_confidence = 0.50
        
    def detect_with_yolo(self, image):
        """Detect using YOLOv8"""
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
                
                # Accept cell phone with reasonable confidence
                if confidence > self.min_yolo_confidence and class_name == 'cell phone':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    
                    # Basic size check
                    if w >= 80 and h >= 80:
                        detections.append({
                            'bbox': (x1, y1, w, h),
                            'confidence': confidence,
                            'method': 'YOLO',
                            'class': class_name
                        })
        
        return detections
    
    def has_screen_characteristics(self, image, bbox):
        """Check if region has display/screen characteristics"""
        x, y, w, h = bbox
        
        # Extract region
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        # Convert to different color spaces
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check 1: Standard deviation (screens have content variation)
        std_dev = np.std(gray)
        has_content = std_dev > 15  # Screens have varying content
        
        # Check 2: Edge density (screens have text/icons)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        has_edges = edge_density > 0.05  # At least 5% edges
        
        # Check 3: Color variety (not uniform like lights/tables)
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        h_variety = np.count_nonzero(h_hist > 10)  # Multiple color hues
        has_colors = h_variety > 5
        
        # Check 4: Brightness (not too bright like lights)
        mean_brightness = np.mean(gray)
        not_too_bright = mean_brightness < 240  # Not a pure light
        
        # Check 5: Not uniform (tables/walls are uniform)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dominant_value = np.max(hist)
        not_uniform = dominant_value < (gray.size * 0.8)  # Not 80% same color
        
        # Score the region
        score = sum([has_content, has_edges, has_colors, not_too_bright, not_uniform])
        
        return score >= 3  # Need at least 3 out of 5 characteristics
    
    def detect_with_contours(self, image):
        """Detect rectangular objects with screen characteristics"""
        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        height, width = image.shape[:2]
        min_area = (height * width) * 0.04
        max_area = (height * width) * 0.85
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Check aspect ratio (phones + partial panels)
                valid_aspect = (0.3 <= aspect_ratio <= 0.9) or (1.1 <= aspect_ratio <= 3.0)
                
                if valid_aspect and w >= 80 and h >= 80:
                    # Check if this looks like a screen/display
                    if self.has_screen_characteristics(image, (x, y, w, h)):
                        # Calculate confidence based on rectangularity
                        perimeter = cv2.arcLength(contour, True)
                        epsilon = 0.02 * perimeter
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        rectangularity = 1.0 - abs(len(approx) - 4) * 0.1
                        confidence = max(0.5, min(0.85, rectangularity))
                        
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'method': 'Contour+Screen',
                            'aspect_ratio': aspect_ratio
                        })
        
        return detections
    
    def merge_detections(self, detections):
        """Merge overlapping detections"""
        if not detections:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
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
            
            inds = np.where(iou <= 0.4)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def detect(self, image):
        """Hybrid detection: YOLO + Smart Contours"""
        all_detections = []
        
        # Method 1: YOLO (preferred)
        yolo_detections = self.detect_with_yolo(image)
        all_detections.extend(yolo_detections)
        
        # Method 2: Contour with screen characteristics (if YOLO finds nothing)
        if len(yolo_detections) == 0:
            contour_detections = self.detect_with_contours(image)
            all_detections.extend(contour_detections)
        
        # Merge overlapping detections
        final_detections = self.merge_detections(all_detections)
        
        return final_detections


def mark_image(image, detections):
    """Mark detected panels on the image"""
    marked = image.copy()
    height, width = marked.shape[:2]
    
    if not detections:
        # No panel detected - show OK
        overlay = marked.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 180, 0), -1)
        marked = cv2.addWeighted(overlay, 0.3, marked, 0.7, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "OK - No Panel Detected"
        font_scale = 1.2
        thickness = 3
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (width - text_width) // 2
        y = 50
        
        cv2.putText(marked, text, (x+2, y+2), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(marked, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # Checkmark
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
            
            # Draw red rectangle
            cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 4)
            
            # Draw corners
            corner_length = 30
            corner_thickness = 6
            cv2.line(marked, (x, y), (x + corner_length, y), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x, y), (x, y + corner_length), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x + w, y), (x + w - corner_length, y), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x + w, y), (x + w, y + corner_length), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x, y + h), (x + corner_length, y + h), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x, y + h), (x, y + h - corner_length), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x + w, y + h), (x + w - corner_length, y + h), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x + w, y + h), (x + w, y + h - corner_length), (0, 0, 255), corner_thickness)
            
            # NG Label
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = "NG"
            font_scale = 2.5
            thickness = 4
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            label_y = max(y - 10, text_height + 20)
            label_x = x
            padding = 15
            
            cv2.rectangle(marked, 
                         (label_x - padding, label_y - text_height - padding),
                         (label_x + text_width + padding, label_y + baseline + padding),
                         (0, 0, 255), -1)
            
            cv2.rectangle(marked, 
                         (label_x - padding, label_y - text_height - padding),
                         (label_x + text_width + padding, label_y + baseline + padding),
                         (255, 255, 255), 2)
            
            cv2.putText(marked, label, (label_x + 2, label_y + 2),
                       font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(marked, label, (label_x, label_y),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Info text
            info_text = f"#{idx} {confidence*100:.1f}% ({method})"
            info_font_scale = 0.6
            info_thickness = 2
            
            (info_width, info_height), _ = cv2.getTextSize(info_text, font, info_font_scale, info_thickness)
            info_y = y + h + 25
            
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
        
        # Warning banner
        overlay = marked.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
        marked = cv2.addWeighted(overlay, 0.3, marked, 0.7, 0)
        
        warning_text = f"PANEL DETECTED - NG ({len(detections)} found)"
        font_scale = 1.0
        thickness = 3
        (text_width, text_height), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = (width - text_width) // 2
        text_y = 50
        
        # Warning X icon
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


detector = AdvancedPanelDetector()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Advanced Mobile Panel Detection API",
        "yolo_loaded": model is not None,
        "version": "2.2 - Hybrid Detection",
        "yolo_confidence": detector.min_yolo_confidence,
        "detection_mode": "hybrid"
    })


@app.route('/detect', methods=['POST'])
def detect_panel():
    """Main detection endpoint"""
    try:
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
        
        # Detect panels
        detections = detector.detect(image)
        
        # Mark the image
        marked_image = mark_image(image, detections)
        
        # Encode
        _, buffer = cv2.imencode('.jpg', marked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        marked_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0.0
        
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
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "detection_mode": "hybrid_yolo_contour"
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
    """Returns marked image directly"""
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
        
        detections = detector.detect(image)
        marked_image = mark_image(image, detections)
        
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
    """Batch processing"""
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
            "results": results
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Mobile Panel Detection API v2.2 - HYBRID")
    print("=" * 60)
    print("✓ YOLO Model:", "Loaded" if model else "Not available")
    print("✓ Detection: YOLO + Smart Contours (screen check)")
    print("✓ YOLO Confidence:", detector.min_yolo_confidence)
    print("✓ Filters out: Lights, tables, walls")
    print("✓ Detects: Full & partial panels")
    print("\n📡 Endpoints:")
    print("  - POST /detect          (JSON)")
    print("  - POST /detect/image    (Image file)")
    print("  - POST /detect/batch    (Multiple)")
    print("  - GET  /health          (Status)")
    print("\n🌐 Server: http://0.0.0.0:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
