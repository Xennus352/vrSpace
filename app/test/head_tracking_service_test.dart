import 'dart:async';
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:web_view/services/head_tracking_service.dart';

Map<String, dynamic> pose(int sequence) => {
  'rotation': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
  'sequence': sequence,
  'sensorSession': 'native-session',
};

void main() {
  test('pose and recenter use viewer host and token automatically', () async {
    final samples = StreamController<dynamic>();
    final requests = <http.Request>[];
    final service = HeadTrackingService(
      'http://10.0.0.2:8080/?token=private',
      samples: samples.stream,
      client: MockClient((request) async {
        requests.add(request);
        return http.Response('{"ok":true}', 200);
      }),
    );
    final connected = service.status.stream.first;
    service.start();
    samples.add(pose(0));
    expect(await connected, 'Head tracking connected');
    expect(
      requests.single.url.toString(),
      'http://10.0.0.2:8080/pose?token=private',
    );
    expect(
      jsonDecode(requests.single.body)['session'],
      endsWith('-native-session'),
    );
    await service.recenter();
    expect(requests.last.url.path, '/control');
    expect(requests.last.url.queryParameters['token'], 'private');
    expect(jsonDecode(requests.last.body), {'action': 'recenter'});
    await service.dispose();
    await samples.close();
  });

  test('slow uploads retain only the newest pose', () async {
    final samples = StreamController<dynamic>();
    final release = Completer<void>();
    final seen = <int>[];
    final sentLatest = Completer<void>();
    final service = HeadTrackingService(
      'http://laptop:8080/?token=t',
      samples: samples.stream,
      client: MockClient((request) async {
        final sequence = jsonDecode(request.body)['sequence'] as int;
        seen.add(sequence);
        if (sequence == 0) await release.future;
        if (sequence == 9) sentLatest.complete();
        return http.Response('{}', 200);
      }),
    );
    service.start();
    samples.add(pose(0));
    await Future<void>.delayed(Duration.zero);
    for (var i = 1; i <= 9; i++) {
      samples.add(pose(i));
    }
    await Future<void>.delayed(Duration.zero);
    release.complete();
    await sentLatest.future;
    expect(seen, [0, 9]);
    await service.dispose();
    await samples.close();
  });
}
