package com.lambrk.scanner.bluetooth

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import java.io.IOException
import java.util.UUID

object BluetoothScannerManager {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private var socket: BluetoothSocket? = null
    private var readJob: Job? = null

    private val _scannedData = MutableSharedFlow<String>()
    val scannedData: SharedFlow<String> = _scannedData

    var connectedDeviceName: String? = null
        private set

    var connectedDeviceAddress: String? = null
        private set

    fun isConnected(): Boolean {
        return socket?.isConnected == true
    }

    @Suppress("MissingPermission")
    fun connect(
        context: Context,
        device: BluetoothDevice,
        onConnected: () -> Unit,
        onError: (String) -> Unit
    ) {
        scope.launch {
            try {
                if (isConnected() && connectedDeviceAddress == device.address) {
                    withContext(Dispatchers.Main) {
                        onConnected()
                    }
                    return@launch
                }

                disconnect()

                val adapter = BluetoothAdapter.getDefaultAdapter()
                if (adapter == null) {
                    withContext(Dispatchers.Main) {
                        onError("Bluetooth is not available.")
                    }
                    return@launch
                }

                if (!adapter.isEnabled) {
                    withContext(Dispatchers.Main) {
                        onError("Turn on Bluetooth.")
                    }
                    return@launch
                }

                if (hasBluetoothScanPermission(context)) {
                    adapter.cancelDiscovery()
                }

                val newSocket = device.createRfcommSocketToServiceRecord(SPP_UUID)
                newSocket.connect()

                socket = newSocket
                connectedDeviceName = device.safeName()
                connectedDeviceAddress = device.address

                startReading(newSocket)

                withContext(Dispatchers.Main) {
                    onConnected()
                }

            } catch (e: Exception) {
                disconnect()

                withContext(Dispatchers.Main) {
                    onError(
                        when (e) {
                            is SecurityException -> "Allow Bluetooth permission."
                            is IOException -> "Could not connect. Make sure scanner is on and paired."
                            else -> e.message ?: "Could not connect."
                        }
                    )
                }
            }
        }
    }

    private fun startReading(socket: BluetoothSocket) {
        readJob?.cancel()

        readJob = scope.launch {
            val buffer = ByteArray(256)
            val line = StringBuilder()

            while (true) {
                val count = try {
                    socket.inputStream.read(buffer)
                } catch (_: IOException) {
                    break
                }

                if (count <= 0) break

                val text = String(buffer, 0, count)

                text.forEach { char ->
                    if (char == '\n' || char == '\r') {
                        val value = line.toString().trim()
                        line.clear()

                        if (value.isNotBlank()) {
                            _scannedData.emit(value)
                        }
                    } else {
                        line.append(char)
                    }
                }
            }

            disconnect()
        }
    }

    fun disconnect() {
        readJob?.cancel()
        readJob = null

        try {
            socket?.close()
        } catch (_: IOException) {
        }

        socket = null
        connectedDeviceName = null
        connectedDeviceAddress = null
    }

    private fun hasBluetoothScanPermission(context: Context): Boolean {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.BLUETOOTH_SCAN
                ) == PackageManager.PERMISSION_GRANTED
    }

    @Suppress("MissingPermission")
    private fun BluetoothDevice.safeName(): String {
        return name?.takeIf { it.isNotBlank() } ?: "Unnamed device"
    }

    private val SPP_UUID: UUID =
        UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
                            }
