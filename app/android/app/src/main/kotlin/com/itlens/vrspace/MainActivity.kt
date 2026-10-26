package com.itlens.vrspace

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.view.Surface
import android.view.WindowManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel

class MainActivity : FlutterActivity(), SensorEventListener {
    private lateinit var sensors: SensorManager
    private var sink: EventChannel.EventSink? = null
    private var sequence = 0L
    private var sensorSession = ""
    private var lastScreen = -1

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        sensors = getSystemService(SENSOR_SERVICE) as SensorManager
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, "vrspace/head_orientation")
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
                    sink = events
                    sequence = 0
                    startOrientation()
                }
                override fun onCancel(arguments: Any?) {
                    sensors.unregisterListener(this@MainActivity)
                    sink = null
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
            })
    }

    private fun startOrientation() {
        val sensor = sensors.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sensors.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        if (sensor == null) {
            sink?.error("NO_ROTATION_SENSOR", "This phone has no fused rotation sensor", null)
            return
        }
        sensors.unregisterListener(this)
        sensorSession = android.os.SystemClock.elapsedRealtimeNanos().toString()
        sensors.registerListener(this, sensor, 20000)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (sink == null) return
        val rotation = FloatArray(9)
        val screenRotation = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(rotation, event.values)
        @Suppress("DEPRECATION")
        val screen = windowManager.defaultDisplay.rotation
        if (screen != lastScreen) {
            sensorSession = android.os.SystemClock.elapsedRealtimeNanos().toString()
            lastScreen = screen
        }
        val axes = when (screen) {
            Surface.ROTATION_90 -> Pair(SensorManager.AXIS_Y, SensorManager.AXIS_MINUS_X)
            Surface.ROTATION_180 -> Pair(SensorManager.AXIS_MINUS_X, SensorManager.AXIS_MINUS_Y)
            Surface.ROTATION_270 -> Pair(SensorManager.AXIS_MINUS_Y, SensorManager.AXIS_X)
            else -> Pair(SensorManager.AXIS_X, SensorManager.AXIS_Y)
        }
        SensorManager.remapCoordinateSystem(rotation, axes.first, axes.second, screenRotation)
        sink?.success(mapOf("rotation" to screenRotation.map { it.toDouble() },
                            "sequence" to sequence++, "sensorSession" to sensorSession))
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    override fun onPause() {
        if (::sensors.isInitialized) sensors.unregisterListener(this)
        super.onPause()
    }
    override fun onResume() {
        super.onResume()
        if (::sensors.isInitialized && sink != null) startOrientation()
    }
}
