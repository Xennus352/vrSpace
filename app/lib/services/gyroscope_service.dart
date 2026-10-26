import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:sensors_plus/sensors_plus.dart';

class GyroReading {
  const GyroReading({
    required this.x,
    required this.y,
    required this.z,
    required this.timestamp,
  });

  final double x;
  final double y;
  final double z;
  final DateTime timestamp;

  Map<String, dynamic> toJson() => <String, dynamic>{
    'x': x,
    'y': y,
    'z': z,
    'timestamp': timestamp.toUtc().toIso8601String(),
  };
}

class GyroscopeService {
  GyroscopeService({
    this.serverUrl,
    this.frameInterval = const Duration(milliseconds: 50),
    this.uploadInterval = const Duration(milliseconds: 100),
    this.samplingPeriod = const Duration(milliseconds: 20),
  });

  final String? serverUrl;
  final Duration frameInterval;
  final Duration uploadInterval;
  final Duration samplingPeriod;

  final StreamController<GyroReading> _frames =
      StreamController<GyroReading>.broadcast();

  StreamSubscription<GyroscopeEvent>? _sensorSubscription;
  Timer? _frameTimer;
  Timer? _uploadTimer;
  GyroReading? _latest;
  bool _uploading = false;

  Stream<GyroReading> get frames => _frames.stream;

  void start() {
    if (_sensorSubscription != null) {
      return;
    }

    _sensorSubscription = gyroscopeEventStream(samplingPeriod: samplingPeriod)
        .listen(
          (GyroscopeEvent event) {
            _latest = GyroReading(
              x: event.x,
              y: event.y,
              z: event.z,
              timestamp: DateTime.now(),
            );
          },
          onError: (Object error) {
            debugPrint('Gyroscope unavailable: $error');
          },
        );

    _frameTimer ??= Timer.periodic(frameInterval, (Timer _) {
      final GyroReading? reading = _latest;
      if (reading != null) {
        _frames.add(reading);
      }
    });

    final String? endpoint = serverUrl?.trim();
    if (endpoint != null && endpoint.isNotEmpty) {
      _uploadTimer ??= Timer.periodic(uploadInterval, (Timer _) {
        unawaited(_upload());
      });
    }
  }

  Future<void> _upload() async {
    final GyroReading? reading = _latest;
    final Uri? endpoint = Uri.tryParse(serverUrl?.trim() ?? '');

    if (reading == null ||
        endpoint == null ||
        !endpoint.hasScheme ||
        endpoint.host.isEmpty ||
        _uploading) {
      return;
    }

    _uploading = true;
    try {
      await http
          .post(
            endpoint,
            headers: const <String, String>{'Content-Type': 'application/json'},
            body: jsonEncode(reading.toJson()),
          )
          .timeout(const Duration(seconds: 5));
    } catch (error) {
      debugPrint('Gyro upload failed: $error');
    } finally {
      _uploading = false;
    }
  }

  Future<void> dispose() async {
    _frameTimer?.cancel();
    _uploadTimer?.cancel();
    await _sensorSubscription?.cancel();
    await _frames.close();
  }
}
