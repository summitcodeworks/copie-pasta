<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Viewer Modal</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .input-section {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            box-sizing: border-box;
        }
        .buttons {
            margin: 10px 0;
        }
        button {
            padding: 10px 20px;
            margin-right: 10px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background: #0056b3;
        }
        button:active {
            background: #004085;
        }
        
        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
        }
        
        .modal.active {
            display: block;
        }
        
        .modal-header {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            padding: 15px 20px;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1001;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .modal-title {
            color: white;
            margin: 0;
            font-size: 18px;
        }
        
        .close-btn {
            background: transparent;
            border: none;
            color: white;
            font-size: 32px;
            cursor: pointer;
            padding: 0;
            width: 40px;
            height: 40px;
            line-height: 32px;
            margin: 0;
            transition: transform 0.2s;
        }
        
        .close-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: rotate(90deg);
        }
        
        .viewer-container {
            width: 100%;
            height: 100%;
            overflow: hidden;
            position: relative;
            cursor: grab;
        }
        
        .viewer-container.dragging {
            cursor: grabbing;
        }
        
        .viewer-container img {
            position: absolute;
            left: 50%;
            top: 50%;
            transform-origin: center center;
            transition: transform 0.1s ease-out;
            user-select: none;
            -webkit-user-drag: none;
            pointer-events: none;
        }
        
        .modal-controls {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1001;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 20px;
            border-radius: 30px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .modal-controls button {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 8px 16px;
            margin: 0;
            font-size: 14px;
        }
        
        .modal-controls button:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        .zoom-info {
            color: white;
            font-size: 14px;
            min-width: 50px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Base64 Image Viewer</h1>
        
        <div class="input-section">
            <label for="base64Input"><strong>Paste Base64 String:</strong></label>
            <textarea id="base64Input" placeholder="Paste your base64 encoded image here (with or without data:image prefix)"></textarea>
        </div>
        
        <div class="buttons">
            <button id="loadBtn">Open in Viewer</button>
            <button id="sampleBtn">Load Sample Image</button>
        </div>
    </div>
    
    <!-- Modal -->
    <div class="modal" id="imageModal">
        <div class="modal-header">
            <h2 class="modal-title">Image Viewer</h2>
            <button class="close-btn" id="closeBtn" title="Close (ESC)">&times;</button>
        </div>
        
        <div class="viewer-container" id="viewerContainer">
            <img id="imageElement" style="display: none;" alt="Loaded image">
        </div>
        
        <div class="modal-controls">
            <button id="zoomOutBtn">-</button>
            <span class="zoom-info" id="zoomInfo">100%</span>
            <button id="zoomInBtn">+</button>
            <button id="resetBtn">Reset</button>
        </div>
    </div>

    <script>
        (function() {
            // State variables
            var scale = 1;
            var posX = 0;
            var posY = 0;
            var isDragging = false;
            var startX = 0;
            var startY = 0;
            var imgWidth = 0;
            var imgHeight = 0;
            
            // DOM elements
            var base64Input = document.getElementById('base64Input');
            var loadBtn = document.getElementById('loadBtn');
            var sampleBtn = document.getElementById('sampleBtn');
            var imageModal = document.getElementById('imageModal');
            var closeBtn = document.getElementById('closeBtn');
            var viewerContainer = document.getElementById('viewerContainer');
            var imageElement = document.getElementById('imageElement');
            var zoomInBtn = document.getElementById('zoomInBtn');
            var zoomOutBtn = document.getElementById('zoomOutBtn');
            var resetBtn = document.getElementById('resetBtn');
            var zoomInfo = document.getElementById('zoomInfo');
            
            // Sample base64 image
            var sampleBase64 = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImciIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPjxzdG9wIG9mZnNldD0iMCUiIHN0eWxlPSJzdG9wLWNvbG9yOiNmZjYzNDc7c3RvcC1vcGFjaXR5OjEiLz48c3RvcCBvZmZzZXQ9IjUwJSIgc3R5bGU9InN0b3AtY29sb3I6IzQ3ZmZhZjtzdG9wLW9wYWNpdHk6MSIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3R5bGU9InN0b3AtY29sb3I6I2E2NDdmZjtzdG9wLW9wYWNpdHk6MSIvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjxyZWN0IHdpZHRoPSI0MDAiIGhlaWdodD0iMzAwIiBmaWxsPSJ1cmwoI2cpIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIzMiIgZmlsbD0id2hpdGUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5TYW1wbGUgSW1hZ2U8L3RleHQ+PC9zdmc+';
            
            // Open modal
            function openModal() {
                imageModal.classList.add('active');
                document.body.style.overflow = 'hidden';
            }
            
            // Close modal
            function closeModal() {
                imageModal.classList.remove('active');
                document.body.style.overflow = '';
            }
            
            // Load image function
            function loadImage() {
                var base64String = base64Input.value.trim();
                
                if (!base64String) {
                    alert('Please paste a base64 string');
                    return;
                }
                
                // Add data URI prefix if not present
                if (base64String.indexOf('data:image') !== 0) {
                    base64String = 'data:image/png;base64,' + base64String;
                }
                
                imageElement.onload = function() {
                    imgWidth = imageElement.naturalWidth;
                    imgHeight = imageElement.naturalHeight;
                    resetView();
                    imageElement.style.display = 'block';
                    openModal();
                };
                
                imageElement.onerror = function() {
                    alert('Failed to load image. Please check your base64 string.');
                    imageElement.style.display = 'none';
                };
                
                imageElement.src = base64String;
            }
            
            // Reset view
            function resetView() {
                scale = 1;
                posX = 0;
                posY = 0;
                updateTransform();
            }
            
            // Update transform
            function updateTransform() {
                imageElement.style.transform = 'translate(-50%, -50%) translate(' + posX + 'px, ' + posY + 'px) scale(' + scale + ')';
                zoomInfo.textContent = Math.round(scale * 100) + '%';
            }
            
            // Zoom functions
            function zoomIn() {
                if (scale < 5) {
                    scale = Math.min(scale * 1.2, 5);
                    updateTransform();
                }
            }
            
            function zoomOut() {
                if (scale > 0.1) {
                    scale = Math.max(scale / 1.2, 0.1);
                    updateTransform();
                }
            }
            
            // Mouse down handler
            function handleMouseDown(e) {
                e.preventDefault();
                if (imageElement.style.display !== 'none') {
                    isDragging = true;
                    startX = e.clientX - posX;
                    startY = e.clientY - posY;
                    viewerContainer.classList.add('dragging');
                }
            }
            
            // Mouse move handler
            function handleMouseMove(e) {
                if (isDragging) {
                    e.preventDefault();
                    posX = e.clientX - startX;
                    posY = e.clientY - startY;
                    updateTransform();
                }
            }
            
            // Mouse up handler
            function handleMouseUp(e) {
                if (isDragging) {
                    e.preventDefault();
                }
                isDragging = false;
                viewerContainer.classList.remove('dragging');
            }
            
            // Wheel zoom handler
            function handleWheel(e) {
                e.preventDefault();
                
                if (e.deltaY < 0) {
                    zoomIn();
                } else {
                    zoomOut();
                }
            }
            
            // Event listeners
            loadBtn.addEventListener('click', loadImage);
            
            sampleBtn.addEventListener('click', function() {
                base64Input.value = sampleBase64;
                loadImage();
            });
            
            closeBtn.addEventListener('click', closeModal);
            
            zoomInBtn.addEventListener('click', zoomIn);
            zoomOutBtn.addEventListener('click', zoomOut);
            resetBtn.addEventListener('click', resetView);
            
            viewerContainer.addEventListener('mousedown', handleMouseDown);
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            viewerContainer.addEventListener('wheel', handleWheel);
            
            // Close modal on background click
            imageModal.addEventListener('click', function(e) {
                if (e.target === imageModal) {
                    closeModal();
                }
            });
            
            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {
                if (imageModal.classList.contains('active')) {
                    if (e.key === 'Escape') {
                        closeModal();
                    } else if (e.key === '+' || e.key === '=') {
                        zoomIn();
                    } else if (e.key === '-' || e.key === '_') {
                        zoomOut();
                    } else if (e.key === '0') {
                        resetView();
                    }
                }
            });
            
            // Load sample on start
            base64Input.value = sampleBase64;
        })();
    </script>
</body>
</html>
