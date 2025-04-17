import 'package:flutter/material.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Generate Music',
      theme: ThemeData(
        fontFamily: 'Montserrat', // Use a modern sans-serif font as in the PDF
        primaryColor: const Color(0xFFEFB7C6), // Pink pastel
        scaffoldBackgroundColor: const Color(0xFFF8F8FF), // Soft background
      ),
      home: const SplashPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class SplashPage extends StatelessWidget {
  const SplashPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background with pastel blobs
          CustomPaint(
            size: MediaQuery.of(context).size,
            painter: PastelBackgroundPainter(),
          ),
          Center(
            child: Container(
              width: 140,
              height: 140,
              decoration: BoxDecoration(
                color: const Color(0xFFEFB7C6), // Pink pastel
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: Colors.white,
                  width: 4,
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.07),
                    blurRadius: 10,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: Center(
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    // The "B" with waveform and pastel overlay
                    Text(
                      'B',
                      style: TextStyle(
                        fontSize: 80,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                        letterSpacing: 2,
                      ),
                    ),
                    Positioned(
                      left: 28,
                      top: 30,
                      child: Container(
                        width: 25,
                        height: 60,
                        decoration: BoxDecoration(
                          color: const Color(0xFFB6F2E5), // Mint pastel
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                    ),
                    // Add a simple waveform as a decorative line
                    Positioned(
                      right: 22,
                      top: 60,
                      child: Container(
                        width: 35,
                        height: 8,
                        decoration: BoxDecoration(
                          color: const Color(0xFFB6F2E5), // Mint pastel
                          borderRadius: BorderRadius.circular(4),
                        ),
                      ),
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

// Painter for pastel blobs background
class PastelBackgroundPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint();

    // Mint blob (top left)
    paint.color = const Color(0xFFB6F2E5);
    canvas.drawOval(
      Rect.fromLTWH(-60, -30, size.width * 0.7, size.height * 0.5),
      paint,
    );

    // Lavender blob (bottom right)
    paint.color = const Color(0xFFD9C6F7);
    canvas.drawOval(
      Rect.fromLTWH(size.width * 0.2, size.height * 0.3, size.width * 0.9, size.height * 0.7),
      paint,
    );

    // White central highlight
    paint.color = Colors.white.withOpacity(0.85);
    canvas.drawOval(
      Rect.fromLTWH(size.width * 0.15, size.height * 0.1, size.width * 0.7, size.height * 0.7),
      paint,
    );
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}