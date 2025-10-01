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
    """Hybrid panel detector - Handles rotated/sideways panels"""
    
    def __init__(self):
        # Lower confidence for better detection of angled panels
        self.min_yolo_confidence = 0.35
        self.min_contour_confidence = 0.45
        
    def detect_with_yolo(self, image):
        """Detect using YOLOv8 - handles all orientations"""
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
                
                # Accept cell phone with lower confidence for angled panels
                if confidence > self.min_yolo_confidence and class_name == 'cell phone':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    
                    # Minimum size check only
                    if w >= 60 and h >= 60:
                        detections.append({
                            'bbox': (x1, y1, w, h),
                            'confidence': confidence,
                            'method': 'YOLO',
                            'class': class_name
                        })
        
        return detections
    
    def get_rotated_rect_bbox(self, rotated_rect):
        """Get bounding box from rotated rectangle"""
        box = cv2.boxPoints(rotated_rect)
        box = np.int0(box)
        x, y, w, h = cv2.boundingRect(box)
        return (x, y, w, h), box
    
    def has_screen_characteristics(self, image, bbox):
        """Check if region has display/screen characteristics - enhanced for rotated panels"""
        x, y, w, h = bbox
        
        # Ensure coordinates are within image bounds
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            return False
        
        # Extract region
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        # Convert to different color spaces
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check 1: Standard deviation (screens have content variation)
        std_dev = np.std(gray)
        has_content = std_dev > 12  # Lowered for rotated panels
        
        # Check 2: Edge density (screens have text/icons)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        has_edges = edge_density > 0.04  # Lowered threshold
        
        # Check 3: Color variety (not uniform like lights/tables)
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        h_variety = np.count_nonzero(h_hist > 8)
        has_colors = h_variety > 4
        
        # Check 4: Brightness (not too bright like lights)
        mean_brightness = np.mean(gray)
        not_too_bright = mean_brightness < 245  # More lenient
        
        # Check 5: Not uniform (tables/walls are uniform)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dominant_value = np.max(hist)
        not_uniform = dominant_value < (gray.size * 0.75)  # More lenient
        
        # Check 6: Texture analysis (screens have texture)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        has_texture = texture_variance > 50
        
        # Score the region (need at least 3 out of 6)
        score = sum([has_content, has_edges, has_colors, not_too_bright, not_uniform, has_texture])
        
        return score >= 3
    
    def detect_with_contours(self, image):
        """Detect rectangular objects including rotated ones"""
        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Multiple edge detection approaches
        edges1 = cv2.Canny(blurred, 30, 100)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        height, width = image.shape[:2]
        min_area = (height * width) * 0.03  # 3% minimum
        max_area = (height * width) * 0.90  # 90% maximum
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Get minimum area rectangle (handles rotation)
                rect = cv2.minAreaRect(contour)
                (cx, cy), (rect_w, rect_h), angle = rect
                
                # Swap w/h if needed to get correct aspect ratio
                if rect_w < rect_h:
                    rect_w, rect_h = rect_h, rect_w
                    angle = angle + 90
                
                # Calculate aspect ratio
                aspect_ratio = float(rect_w) / rect_h if rect_h > 0 else 0
                
                # Very lenient aspect ratio for any orientation
                # Allows square-ish, portrait, landscape, and everything in between
                valid_aspect = 0.25 <= aspect_ratio <= 4.0
                
                if valid_aspect and rect_w >= 60 and rect_h >= 60:
                    # Get bounding box for the rotated rectangle
                    bbox, box_points = self.get_rotated_rect_bbox(rect)
                    
                    # Check if this looks like a screen/display
                    if self.has_screen_characteristics(image, bbox):
                        # Calculate confidence
                        perimeter = cv2.arcLength(contour, True)
                        epsilon = 0.02 * perimeter
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        rectangularity = 1.0 - abs(len(approx) - 4) * 0.08
                        confidence = max(0.45, min(0.85, rectangularity))
                        
                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'method': f'Contour+Screen (angle:{int(angle)}°)',
                            'aspect_ratio': aspect_ratio,
                            'angle': angle,
                            'rotated_rect': box_points
                        })
        
        return detections
    
    def detect_with_color(self, image):
        """Detect based on screen colors - works for any orientation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Multiple color ranges for screens
        masks = []
        
        # Range 1: Blue screens
        masks.append(cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255])))
        
        # Range 2: White/bright screens
        masks.append(cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255])))
        
        # Range 3: Cyan/Light blue
        masks.append(cv2.inRange(hsv, np.array([80, 30, 100]), np.array([100, 255, 255])))
        
        # Combine all masks
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        height, width = image.shape[:2]
        min_area = (height * width) * 0.025
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Use minAreaRect for rotation handling
                rect = cv2.minAreaRect(contour)
                bbox, box_points = self.get_rotated_rect_bbox(rect)
                x, y, w, h = bbox
                
                if w >= 60 and h >= 60:
                    confidence = min(0.75, area / (height * width * 0.5))
                    detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'method': 'Color',
                        'rotated_rect': box_points
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
            
            inds = np.where(iou <= 0.35)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def detect(self, image):
        """Hybrid detection: YOLO + Contours + Color for all orientations"""
        all_detections = []
        
        # Method 1: YOLO (works for rotated objects)
        yolo_detections = self.detect_with_yolo(image)
        all_detections.extend(yolo_detections)
        
        # Method 2: Contour with rotation support
        contour_detections = self.detect_with_contours(image)
        all_detections.extend(contour_detections)
        
        # Method 3: Color-based (orientation independent)
        color_detections = self.detect_with_color(image)
        all_detections.extend(color_detections)
        
        # Merge overlapping detections
        final_detections = self.merge_detections(all_detections)
        
        return final_detections


def mark_image(image, detections):
    """Mark detected panels - handles rotated panels"""
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
            
            # If we have rotated rect points, draw them too
            if 'rotated_rect' in detection:
                box_points = detection['rotated_rect']
                cv2.drawContours(marked, [box_points], 0, (0, 100, 255), 2)  # Blue outline for rotation
            
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
            
            # Info text with angle if available
            angle_info = ""
            if 'angle' in detection:
                angle_info = f" {int(detection['angle'])}°"
            info_text = f"#{idx} {confidence*100:.1f}%{angle_info} ({method})"
            info_font_scale = 0.5
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
        "version": "2.3 - Rotation Support",
        "yolo_confidence": detector.min_yolo_confidence,
        "detection_mode": "hybrid_with_rotation",
        "supports": "all orientations including sideways/rotated panels"
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
        
        # Detect panels (all orientations)
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
                    "angle": round(d.get('angle', 0), 1) if 'angle' in d else None,
                    "area": d['bbox'][2] * d['bbox'][3]
                }
                for idx, d in enumerate(detections)
            ],
            "marked_image": f"data:image/jpeg;base64,{marked_image_base64}",
            "message": f"Detected {len(detections)} mobile panel(s)" if detections else "No mobile panels detected",
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "detection_mode": "hybrid_rotation_support"
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
    print("=" * 70)
    print("🚀 Mobile Panel Detection API v2.3 - ROTATION SUPPORT")
    print("=" * 70)
    print("✓ YOLO Model:", "Loaded" if model else "Not available")
    print("✓ Detection: YOLO + Rotated Contours + Color")
    print("✓ YOLO Confidence:", detector.min_yolo_confidence)
    print("✓ Supports: ALL orientations (0°, 45°, 90°, 180°, 270°, etc.)")
    print("✓ Filters: Lights, tables, walls")
    print("✓ Detects: Full, partial, sideways, rotated panels")
    print("\n📡 Endpoints:")
    print("  - POST /detect          (JSON)")
    print("  - POST /detect/image    (Image file)")
    print("  - POST /detect/batch    (Multiple)")
    print("  - GET  /health          (Status)")
    print("\n🌐 Server: http://0.0.0.0:5000")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
