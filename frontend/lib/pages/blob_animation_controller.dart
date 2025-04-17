import 'package:flutter/material.dart';

class BlobAnimationController {
  final AnimationController controller;

  BlobAnimationController({
    required TickerProvider vsync,
    Duration duration = const Duration(seconds: 8),
  }) : controller = AnimationController(
          vsync: vsync,
          duration: duration,
        );

  void setDuration(Duration duration) {
    controller.duration = duration;
  }

  void start() {
    controller.repeat();
  }

  void dispose() {
    controller.dispose();
  }
}