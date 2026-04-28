Below is a clean Kotlin permission setup for Android 8 API 26 and above.

It handles:

Camera
Microphone audio recording
Notifications
Image / video / audio media read access
Old read/write storage for Android 8 and 9

Android 13 introduced POST_NOTIFICATIONS, Android 13 also replaced READ_EXTERNAL_STORAGE with READ_MEDIA_IMAGES, READ_MEDIA_VIDEO, and READ_MEDIA_AUDIO. Android 14 added selected photo/video access through READ_MEDIA_VISUAL_USER_SELECTED. 


---

1. AndroidManifest.xml

<!-- Camera -->
<uses-permission android:name="android.permission.CAMERA" />

<!-- Microphone / audio recording -->
<uses-permission android:name="android.permission.RECORD_AUDIO" />

<!-- Notification permission, required from Android 13+ -->
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />

<!-- Android 8 to Android 12L: read media/storage -->
<uses-permission
    android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32" />

<!-- Android 8 and Android 9 only: write external storage -->
<uses-permission
    android:name="android.permission.WRITE_EXTERNAL_STORAGE"
    android:maxSdkVersion="28" />

<!-- Android 13+: read images, videos, audio -->
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
<uses-permission android:name="android.permission.READ_MEDIA_VIDEO" />
<uses-permission android:name="android.permission.READ_MEDIA_AUDIO" />

<!-- Android 14+: selected photo/video access -->
<uses-permission android:name="android.permission.READ_MEDIA_VISUAL_USER_SELECTED" />

Important: for Android 10 and above, avoid old-style file path writing. Use MediaStore or app-specific storage. Android docs recommend MediaStore for shared media files, and Android 11 scoped storage restricts broad file access. 


---

2. Kotlin permission helper

Use this in your Activity.

import android.Manifest
import android.os.Build
import android.os.Bundle
import android.content.pm.PackageManager
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->

            val deniedPermissions = result.filterValues { granted ->
                !granted
            }.keys

            if (deniedPermissions.isEmpty()) {
                // All permissions granted
                onAllPermissionsGranted()
            } else {
                // Some permissions denied
                onPermissionsDenied(deniedPermissions)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        requestRequiredPermissions()
    }

    private fun requestRequiredPermissions() {
        val permissions = mutableListOf<String>()

        // Camera
        permissions.add(Manifest.permission.CAMERA)

        // Microphone recording
        permissions.add(Manifest.permission.RECORD_AUDIO)

        // Notifications, Android 13+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        // Media / storage permissions
        when {
            // Android 14+
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE -> {
                permissions.add(Manifest.permission.READ_MEDIA_IMAGES)
                permissions.add(Manifest.permission.READ_MEDIA_VIDEO)
                permissions.add(Manifest.permission.READ_MEDIA_AUDIO)

                // Allows selected photo/video access on Android 14+
                permissions.add(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)
            }

            // Android 13
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU -> {
                permissions.add(Manifest.permission.READ_MEDIA_IMAGES)
                permissions.add(Manifest.permission.READ_MEDIA_VIDEO)
                permissions.add(Manifest.permission.READ_MEDIA_AUDIO)
            }

            // Android 8 to Android 12L
            else -> {
                permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)

                // WRITE_EXTERNAL_STORAGE is useful only till Android 9
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }
        }

        val permissionsToRequest = permissions.filter { permission ->
            ContextCompat.checkSelfPermission(
                this,
                permission
            ) != PackageManager.PERMISSION_GRANTED
        }

        if (permissionsToRequest.isNotEmpty()) {
            permissionLauncher.launch(permissionsToRequest.toTypedArray())
        } else {
            onAllPermissionsGranted()
        }
    }

    private fun onAllPermissionsGranted() {
        // Continue your work here
        // Example: open camera, start recording, read media files, etc.
    }

    private fun onPermissionsDenied(deniedPermissions: Set<String>) {
        // Handle denied permissions here
        deniedPermissions.forEach {
            println("Denied permission: $it")
        }
    }
}


---

3. Version-wise permission logic

Android version	API	Permission needed

Android 8, 8.1, 9	26 to 28	READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, CAMERA, RECORD_AUDIO
Android 10	29	Prefer MediaStore; old write storage is limited
Android 11, 12, 12L	30 to 32	READ_EXTERNAL_STORAGE; use MediaStore for writing shared media
Android 13	33	READ_MEDIA_IMAGES, READ_MEDIA_VIDEO, READ_MEDIA_AUDIO, POST_NOTIFICATIONS
Android 14+	34+	Same as Android 13, plus possible selected media access with READ_MEDIA_VISUAL_USER_SELECTED



---

4. Simple version without Android 14 selected photo access

Use this if you do not want selected photo/video handling yet.

private fun getRequiredPermissions(): Array<String> {
    val permissions = mutableListOf<String>()

    permissions.add(Manifest.permission.CAMERA)
    permissions.add(Manifest.permission.RECORD_AUDIO)

    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        permissions.add(Manifest.permission.READ_MEDIA_IMAGES)
        permissions.add(Manifest.permission.READ_MEDIA_VIDEO)
        permissions.add(Manifest.permission.READ_MEDIA_AUDIO)
    } else {
        permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)

        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
    }

    return permissions.toTypedArray()
}

Then request:

private val permissionLauncher =
    registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
        val allGranted = result.values.all { it }

        if (allGranted) {
            onAllPermissionsGranted()
        } else {
            onPermissionsDenied(result.filterValues { !it }.keys)
        }
    }

private fun requestPermissionsNow() {
    val permissionsToRequest = getRequiredPermissions().filter { permission ->
        ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED
    }

    if (permissionsToRequest.isNotEmpty()) {
        permissionLauncher.launch(permissionsToRequest.toTypedArray())
    } else {
        onAllPermissionsGranted()
    }
}


---

Important note for storage write

For Android 10 and above, do not depend on WRITE_EXTERNAL_STORAGE.

Use:

MediaStore.Images.Media.EXTERNAL_CONTENT_URI
MediaStore.Video.Media.EXTERNAL_CONTENT_URI
MediaStore.Audio.Media.EXTERNAL_CONTENT_URI

For saving your own app files, use:

getExternalFilesDir(null)

For full file manager type access, Android has MANAGE_EXTERNAL_STORAGE, but Google Play allows it only for limited file-management use cases. It gives broad read/write access to shared storage, so avoid it unless your app truly needs all-files access. 
