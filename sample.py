<!-- Add these declarations inside the <manifest> tag in your AndroidManifest.xml file.
     Place them before the <application> tag for clarity.
     These cover all permissions from the PermissionManager and CameraX integration:
     - Camera and Audio: Always needed.
     - Notifications: For API 33+.
     - Storage: Legacy for pre-API 29; granular for API 33+ shared media access.
     - Camera feature: Optional, but recommended for CameraX to handle devices without camera gracefully.

     Note: For app-private storage (e.g., getExternalFilesDir), no runtime permissions needed on API 29+.
     Adjust based on your exact needs (e.g., add READ_MEDIA_VIDEO if accessing videos).
-->

<!-- Camera Permission -->
<uses-permission android:name="android.permission.CAMERA" />

<!-- Audio (Microphone) Permission -->
<uses-permission android:name="android.permission.RECORD_AUDIO" />

<!-- Notification Permission (API 33+) -->
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />

<!-- Legacy Storage Permissions (only for API < 29; ignored later) -->
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="28" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"
    android:maxSdkVersion="28" />

<!-- Granular Media Permissions for Android 13+ (API 33+) - For shared media access -->
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
<!-- Add if needed: <uses-permission android:name="android.permission.READ_MEDIA_VIDEO" /> -->
<!-- Add if needed: <uses-permission android:name="android.permission.READ_MEDIA_AUDIO" /> -->

<!-- Camera Feature (for CameraX; set required="false" to support non-camera devices) -->
<uses-feature android:name="android.hardware.camera"
    android:required="false" />
<uses-feature android:name="android.hardware.camera.autofocus"
    android:required="false" />

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

/**
 * Updated PermissionManager to handle storage permissions correctly across API levels.
 * Key Changes for Android 13+ (API 33+):
 * - READ_EXTERNAL_STORAGE and WRITE_EXTERNAL_STORAGE are deprecated and ineffective.
 * - For app-specific storage (e.g., getExternalFilesDir), no runtime permissions needed on API 29+.
 * - For shared media access on API 33+, use granular READ_MEDIA_* permissions (e.g., READ_MEDIA_IMAGES for photos).
 * - This version conditionally requests appropriate permissions based on Build.VERSION.SDK_INT.
 * - If only app-private access is needed (as in CameraX example), storage permissions can be skipped entirely.
 *
 * Usage remains the same; it auto-adapts.
 */
class PermissionManager(
    private val activity: AppCompatActivity,
    private val context: Context = activity
) {

    // Launcher for multiple permissions
    private val multiplePermissionLauncher: ActivityResultLauncher<Array<String>> =
        activity.registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            val granted = permissions.filterValues { it }.keys.toSet()
            val denied = permissions.filterValues { !it }.keys.toSet()
            allPermissionsCallback?.invoke(granted, denied)
            allPermissionsCallback = null
        }

    // Launcher for single permissions (used for notifications)
    private val singlePermissionLauncher: ActivityResultLauncher<String> =
        activity.registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            notificationCallback?.invoke(isGranted)
            notificationCallback = null
        }

    // Callbacks
    private var allPermissionsCallback: ((Set<String>, Set<String>) -> Unit)? = null
    private var notificationCallback: ((Boolean) -> Unit)? = null

    // Updated Permission groups with API-level awareness
    companion object {
        val NOTIFICATION_PERMISSIONS = arrayOf(Manifest.permission.POST_NOTIFICATIONS)
        val CAMERA_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        val AUDIO_PERMISSIONS = arrayOf(Manifest.permission.RECORD_AUDIO)

        // Conditional storage permissions
        val STORAGE_PERMISSIONS: Array<String>
            get() {
                return when {
                    Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU -> {
                        // Android 13+: Granular media permissions (request only if shared media access needed)
                        // For photos: READ_MEDIA_IMAGES; adjust based on use case
                        arrayOf(Manifest.permission.READ_MEDIA_IMAGES)
                    }
                    Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q -> {
                        // Android 10-12: No runtime permission for app-specific; skip if not accessing shared
                        emptyArray() // Or add if needed for legacy shared access
                    }
                    else -> {
                        // Pre-Android 10: Legacy storage
                        arrayOf(
                            Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE
                        )
                    }
                }
            }

        // All combined (auto-adapts)
        val ALL_PERMISSIONS: Array<String>
            get() = NOTIFICATION_PERMISSIONS + STORAGE_PERMISSIONS + CAMERA_PERMISSIONS + AUDIO_PERMISSIONS
    }

    /**
     * Requests all permissions (adapts storage based on API level) and invokes callback immediately.
     * For Android 13+, storage request is minimal/empty if only app-private needed.
     */
    fun requestAllPermissions(callback: (granted: Set<String>, denied: Set<String>) -> Unit) {
        val permissionsToRequest = ALL_PERMISSIONS.filter { !isPermissionGranted(it) }.toTypedArray()
        if (permissionsToRequest.isEmpty()) {
            callback(emptySet(), emptySet())
            return
        }
        allPermissionsCallback = callback
        multiplePermissionLauncher.launch(permissionsToRequest)
    }

    /**
     * Updated: Requests storage permissions conditionally.
     * On Android 13+, requests READ_MEDIA_IMAGES (for photo access); empty on API 29-32 for app-private.
     */
    fun requestStoragePermissions(callback: (granted: Set<String>, denied: Set<String>) -> Unit) {
        val permissionsToRequest = STORAGE_PERMISSIONS.filter { !isPermissionGranted(it) }.toTypedArray()
        if (permissionsToRequest.isEmpty()) {
            // All "granted" by default for app-private on modern APIs
            callback(STORAGE_PERMISSIONS.toSet(), emptySet())
            return
        }
        allPermissionsCallback = { granted, denied ->
            val storageGranted = granted.intersect(STORAGE_PERMISSIONS.toSet())
            val storageDenied = denied.intersect(STORAGE_PERMISSIONS.toSet())
            callback(storageGranted, storageDenied)
        }
        multiplePermissionLauncher.launch(permissionsToRequest)
    }

    // Other methods unchanged (requestNotificationPermission, requestCameraPermission, etc.)
    fun requestNotificationPermission(callback: (Boolean) -> Unit) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU || isPermissionGranted(Manifest.permission.POST_NOTIFICATIONS)) {
            callback(true)
            return
        }
        notificationCallback = callback
        singlePermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    fun requestCameraPermission(callback: (Boolean) -> Unit) {
        if (isPermissionGranted(Manifest.permission.CAMERA)) {
            callback(true)
            return
        }
        allPermissionsCallback = { granted, _ ->
            callback(granted.contains(Manifest.permission.CAMERA))
        }
        multiplePermissionLauncher.launch(CAMERA_PERMISSIONS)
    }

    fun requestAudioPermission(callback: (Boolean) -> Unit) {
        if (isPermissionGranted(Manifest.permission.RECORD_AUDIO)) {
            callback(true)
            return
        }
        allPermissionsCallback = { granted, _ ->
            callback(granted.contains(Manifest.permission.RECORD_AUDIO))
        }
        multiplePermissionLauncher.launch(AUDIO_PERMISSIONS)
    }

    private fun isPermissionGranted(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
    }

    fun areAllPermissionsGranted(permissions: Array<String>): Boolean {
        return permissions.all { isPermissionGranted(it) }
    }

    // New helper: Check if storage access is "granted" (considering API levels)
    fun isStorageAccessGranted(): Boolean {
        return when {
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q -> true // App-private always allowed
            else -> STORAGE_PERMISSIONS.all { isPermissionGranted(it) }
        }
    }
}