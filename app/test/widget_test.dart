import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:web_view/main.dart';

void main() {
  testWidgets('shows URL entry form', (WidgetTester tester) async {
    await tester.pumpWidget(const GyroWebViewApp());

    expect(find.text('VrSpace'), findsOneWidget);
    expect(find.text('Open full screen'), findsOneWidget);
    expect(find.byType(TextFormField), findsOneWidget);
    expect(find.text('vrSpace phone URL'), findsOneWidget);
  });

  testWidgets('rejects an invalid URL', (WidgetTester tester) async {
    await tester.pumpWidget(const GyroWebViewApp());

    await tester.tap(find.text('Open full screen'));
    await tester.pump();

    expect(find.text('Enter a valid http or https URL'), findsOneWidget);
  });
}
