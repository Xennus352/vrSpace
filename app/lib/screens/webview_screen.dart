import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webview_flutter/webview_flutter.dart';

import '../services/head_tracking_service.dart';

class WebViewScreen extends StatefulWidget {
  const WebViewScreen({super.key, required this.url, this.serverUrl});

  final String url;
  final String? serverUrl;

  @override
  State<WebViewScreen> createState() => _WebViewScreenState();
}

class _WebViewScreenState extends State<WebViewScreen> {
  late final WebViewController _controller;
  late final HeadTrackingService _headTracking;
  StreamSubscription<String>? _trackingSubscription;
  String _trackingStatus = 'Waiting for head sensor';
  bool _showHud = true;

  bool _isLoading = true;
  int _progress = 0;
  String? _loadError;

  @override
  void initState() {
    super.initState();

    SystemChrome.setPreferredOrientations(const <DeviceOrientation>[
      DeviceOrientation.landscapeLeft,
    ]);
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);

    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(Colors.black)
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageStarted: (String url) {
            if (mounted) {
              setState(() {
                _isLoading = true;
                _progress = 0;
                _loadError = null;
              });
            }
          },
          onProgress: (int progress) {
            if (mounted) {
              setState(() => _progress = progress);
            }
          },
          onPageFinished: (String url) {
            if (mounted) {
              setState(() {
                _isLoading = false;
                _progress = 100;
              });
            }
          },
          onWebResourceError: (WebResourceError error) {
            if (error.isForMainFrame ?? true) {
              if (mounted) {
                setState(() {
                  _isLoading = false;
                  _loadError = error.description;
                });
              }
            }
          },
        ),
      )
      ..loadRequest(Uri.parse(widget.url));

    _headTracking = HeadTrackingService(widget.url);
    _trackingSubscription = _headTracking.status.stream.listen((String status) {
      if (mounted && _trackingStatus != status) {
        setState(() => _trackingStatus = status);
      }
    });
    _headTracking.start();
  }

  Future<void> _reload() async {
    setState(() {
      _isLoading = true;
      _loadError = null;
      _progress = 0;
    });
    await _controller.reload();
  }

  void _exit() {
    Navigator.of(context).pop();
  }

  @override
  void dispose() {
    _trackingSubscription?.cancel();
    unawaited(_headTracking.dispose());
    SystemChrome.setPreferredOrientations(DeviceOrientation.values);
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: <Widget>[
          WebViewWidget(controller: _controller),
          if (_isLoading)
            Align(
              alignment: Alignment.topCenter,
              child: SafeArea(
                bottom: false,
                child: Padding(
                  padding: const EdgeInsets.only(top: 64, left: 16, right: 16),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: _progress > 0 && _progress < 100
                          ? _progress / 100
                          : null,
                      minHeight: 3,
                      backgroundColor: Colors.white.withValues(alpha: 0.15),
                    ),
                  ),
                ),
              ),
            ),
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: SafeArea(
              minimum: const EdgeInsets.all(12),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  _HudButton(
                    icon: _showHud ? Icons.visibility_off : Icons.menu,
                    tooltip: _showHud
                        ? 'Hide app controls'
                        : 'Show app controls',
                    onTap: () => setState(() => _showHud = !_showHud),
                  ),
                  if (_showHud) ...<Widget>[
                    const SizedBox(width: 10),
                    _HudButton(
                      icon: Icons.arrow_back_rounded,
                      tooltip: 'Back to URL',
                      onTap: _exit,
                    ),
                    const SizedBox(width: 10),
                    _HudButton(
                      icon: Icons.refresh_rounded,
                      tooltip: 'Reload page',
                      onTap: _reload,
                    ),
                    const SizedBox(width: 10),
                    _HudButton(
                      icon: Icons.center_focus_strong,
                      tooltip: 'Recenter head',
                      onTap: () async {
                        try {
                          await _headTracking.recenter();
                        } catch (_) {
                          if (mounted) {
                            setState(
                              () => _trackingStatus =
                                  'Recenter failed: check connection',
                            );
                          }
                        }
                      },
                    ),
                    const Spacer(),
                    Flexible(
                      child: Text(
                        _trackingStatus,
                        style: const TextStyle(
                          color: Colors.white,
                          backgroundColor: Colors.black54,
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
          if (_loadError != null)
            Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 440),
                child: Container(
                  margin: const EdgeInsets.all(24),
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: const Color(0xFF141B30),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: Colors.white.withValues(alpha: 0.1),
                    ),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Icon(
                        Icons.wifi_off_rounded,
                        size: 40,
                        color: Theme.of(context).colorScheme.error,
                      ),
                      const SizedBox(height: 14),
                      Text(
                        'Could not load page',
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _loadError!,
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.6),
                        ),
                      ),
                      const SizedBox(height: 20),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: <Widget>[
                          OutlinedButton(
                            onPressed: _exit,
                            child: const Text('Change URL'),
                          ),
                          const SizedBox(width: 12),
                          FilledButton(
                            onPressed: _reload,
                            child: const Text('Retry'),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _HudButton extends StatelessWidget {
  const _HudButton({
    required this.icon,
    required this.tooltip,
    required this.onTap,
  });

  final IconData icon;
  final String tooltip;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      child: Material(
        color: Colors.black.withValues(alpha: 0.55),
        shape: const CircleBorder(),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(10),
            child: Icon(icon, color: Colors.white, size: 22),
          ),
        ),
      ),
    );
  }
}
