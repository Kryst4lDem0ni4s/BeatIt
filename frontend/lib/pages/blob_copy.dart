import 'package:flutter/material.dart';

class Blob extends StatelessWidget {
  final double width;
  final double height;
  final Color color;
  final double opacity;
  
  const Blob({
    super.key,
    required this.width,
    required this.height,
    required this.color,
    this.opacity = 1.0,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: width,
      height: height,
      child: CustomPaint(
        painter: BlobPainter(color.withOpacity(opacity)),
      ),
    );
  }
}

class BlobPainter extends CustomPainter {
  final Color color;
  BlobPainter(this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = color;
    final path = Path();
    path.moveTo(size.width * 0.2, 0);
    path.quadraticBezierTo(
        size.width * 0.9, size.height * 0.1, size.width * 0.8, size.height * 0.7);
    path.quadraticBezierTo(
        size.width * 0.7, size.height * 1.1, size.width * 0.1, size.height * 0.9);
    path.quadraticBezierTo(
        0, size.height * 0.7, size.width * 0.2, 0);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
