import 'package:flutter/material.dart';

import 'screens/url_entry_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const GyroWebViewApp());
}

class GyroWebViewApp extends StatelessWidget {
  const GyroWebViewApp({super.key});

  @override
  Widget build(BuildContext context) {
    final ColorScheme colorScheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF00E5A0),
      brightness: Brightness.dark,
    );

    return MaterialApp(
      title: 'VrSpace',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: colorScheme,
        scaffoldBackgroundColor: const Color(0xFF0B1020),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: const Color(0xFF141B30),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide.none,
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide(color: Colors.white.withValues(alpha: 0.08)),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide(color: colorScheme.primary, width: 1.5),
          ),
        ),
      ),
      home: const UrlEntryScreen(),
    );
  }
}
