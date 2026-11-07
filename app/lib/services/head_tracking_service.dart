import 'dart:async';
import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

/// Native fused orientation -> the same authenticated server as the video.
class HeadTrackingService {
  HeadTrackingService(String viewerUrl, {http.Client? client, this.samples})
    : endpoint = Uri.parse(viewerUrl).removeFragment().replace(path: '/pose'),
      _client = client ?? http.Client();

  final Uri endpoint;
  final http.Client _client;
  final Stream<dynamic>? samples;
  static const _sensor = EventChannel('vrspace/head_orientation');
  final _session = DateTime.now().microsecondsSinceEpoch.toString();
  StreamSubscription<dynamic>? _subscription;
  Map<String, dynamic>? _latest;
  bool _sending = false;
  bool _disposed = false;
  final status = StreamController<String>.broadcast();

  void start() {
    _subscription ??= (samples ?? _sensor.receiveBroadcastStream()).listen(
      (dynamic event) {
        final data = Map<String, dynamic>.from(event as Map);
        _latest = <String, dynamic>{
          ...data,
          'session': '$_session-${data['sensorSession']}',
        };
        unawaited(_send());
      },
      onError: (Object error) {
        if (!_disposed) status.add('Head sensor unavailable');
      },
    );
  }

  Future<void> _send() async {
    if (_sending || _disposed) return;
    _sending = true;
    try {
      // Keep at most one latest pose while a request is in flight.
      while (_latest != null && !_disposed) {
        final data = _latest;
        _latest = null;
        try {
          final response = await _client
              .post(
                endpoint,
                headers: const {'Content-Type': 'application/json'},
                body: jsonEncode(data),
              )
              .timeout(const Duration(milliseconds: 600));
          if (!_disposed) {
            status.add(
              response.statusCode == 200
                  ? 'Head tracking connected'
                  : 'Head tracking rejected (${response.statusCode})',
            );
          }
        } catch (_) {
          if (!_disposed) status.add('Head tracking disconnected');
          // Wait for another sensor sample rather than retrying a backlog.
          _latest = null;
          break;
        }
      }
    } finally {
      _sending = false;
    }
  }

  Future<void> recenter() async {
    final response = await _client
        .post(
          endpoint.replace(path: '/control'),
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({'action': 'recenter'}),
        )
        .timeout(const Duration(seconds: 2));
    if (response.statusCode != 200) throw StateError('Recenter rejected');
  }

  Future<void> dispose() async {
    _disposed = true;
    await _subscription?.cancel();
    _client.close();
    await status.close();
  }
}
