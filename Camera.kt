@Composable
private fun CameraPreview(onQrDetected: (String) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember { Executors.newSingleThreadExecutor() }

    AndroidView(
        factory = { ctx ->

            val previewView = PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }

            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

            cameraProviderFuture.addListener({

                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .build()
                    .also {
                        it.setAnalyzer(
                            executor,
                            QrCodeAnalyzer(onQrDetected)
                        )
                    }

                try {
                    cameraProvider.unbindAll()

                    val camera = cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageAnalysis
                    )

                    val cameraControl = camera.cameraControl
                    val cameraInfo = camera.cameraInfo

                    val scaleGestureDetector =
                        android.view.ScaleGestureDetector(
                            ctx,
                            object : android.view.ScaleGestureDetector.SimpleOnScaleGestureListener() {

                                override fun onScale(
                                    detector: android.view.ScaleGestureDetector
                                ): Boolean {

                                    val currentZoom =
                                        cameraInfo.zoomState.value?.zoomRatio ?: 1f

                                    val newZoom =
                                        currentZoom * detector.scaleFactor

                                    val minZoom =
                                        cameraInfo.zoomState.value?.minZoomRatio ?: 1f

                                    val maxZoom =
                                        cameraInfo.zoomState.value?.maxZoomRatio ?: 10f

                                    cameraControl.setZoomRatio(
                                        newZoom.coerceIn(minZoom, maxZoom)
                                    )

                                    return true
                                }
                            }
                        )

                    previewView.setOnTouchListener { _, event ->
                        scaleGestureDetector.onTouchEvent(event)
                        true
                    }

                } catch (exc: Exception) {
                    exc.printStackTrace()
                }

            }, ContextCompat.getMainExecutor(ctx))

            previewView
        },
        modifier = Modifier.fillMaxSize()
    )
}
