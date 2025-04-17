import 'package:flutter/material.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Generate Music',
      theme: ThemeData(
        fontFamily: 'Montserrat',
        scaffoldBackgroundColor: const Color(0xFF0A0726),
        primaryColor: const Color(0xFFFF4CA0),
        textTheme: const TextTheme(
          bodyMedium: TextStyle(color: Colors.white),
        ),
      ),
      home: const SplashSignInPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class SplashSignInPage extends StatelessWidget {
  const SplashSignInPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 350),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 36),
          decoration: BoxDecoration(
            color: const Color(0xFF0A0726),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Logo
              Container(
                width: 60,
                height: 60,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  color: Colors.white12,
                  // Replace with your logo asset
                  image: const DecorationImage(
                    image: AssetImage('assets/logo.png'),
                    fit: BoxFit.contain,
                  ),
                ),
              ),
              const SizedBox(height: 32),
              // Waveform background with text
              Stack(
                alignment: Alignment.center,
                children: [
                  // Simulated waveform using bars
                  Positioned.fill(
                    child: CustomPaint(
                      painter: WaveformPainter(),
                    ),
                  ),
                  Column(
                    children: const [
                      Text(
                        "Let's Get",
                        style: TextStyle(
                          fontSize: 34,
                          fontWeight: FontWeight.w400,
                          color: Color(0xFFFF4CA0),
                          letterSpacing: 1.2,
                        ),
                      ),
                      SizedBox(height: 2),
                      Text(
                        'Started!',
                        style: TextStyle(
                          fontSize: 38,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                          letterSpacing: 1.2,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 38),
              // Sign In Button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFFF4CA0),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                  ),
                  onPressed: () {},
                  child: const Text(
                    'SIGN IN',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2,
                      fontSize: 18,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 18),
              // Or sign in with
              const Text(
                'OR SIGN IN WITH',
                style: TextStyle(
                  color: Colors.white60,
                  fontSize: 13,
                  letterSpacing: 1.1,
                ),
              ),
              const SizedBox(height: 12),
              // Social Icons
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SocialButton(
                    icon: Icons.mail_outline,
                    onTap: () {},
                  ),
                  const SizedBox(width: 18),
                  SocialButton(
                    icon: Icons.phone_iphone,
                    onTap: () {},
                  ),
                ],
              ),
              const SizedBox(height: 32),
              // Bottom text
              Column(
                children: const [
                  Text(
                    "DIDN'T HAVE ACCOUNT?",
                    style: TextStyle(
                      color: Colors.white54,
                      fontSize: 13,
                      letterSpacing: 1.1,
                    ),
                  ),
                  SizedBox(height: 2),
                  Text(
                    "SIGN UP NOW",
                    style: TextStyle(
                      color: Color(0xFFFF4CA0),
                      fontWeight: FontWeight.bold,
                      fontSize: 13,
                      letterSpacing: 1.1,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// Social Button Widget
class SocialButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  const SocialButton({super.key, required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Ink(
      decoration: BoxDecoration(
        color: Colors.white10,
        shape: BoxShape.circle,
        border: Border.all(color: Colors.white24, width: 1.5),
      ),
      child: IconButton(
        icon: Icon(icon, color: Colors.white, size: 26),
        onPressed: onTap,
      ),
    );
  }
}

// Waveform Painter
class WaveformPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final barWidth = size.width / 18;
    final paint = Paint()
      ..color = const Color(0xFFFF4CA0).withOpacity(0.5)
      ..strokeCap = StrokeCap.round;

    final heights = [
      18.0, 32.0, 22.0, 44.0, 30.0, 54.0, 40.0, 62.0, 48.0,
      62.0, 40.0, 54.0, 30.0, 44.0, 22.0, 32.0, 18.0
    ];

    for (int i = 0; i < heights.length; i++) {
      final x = i * barWidth + barWidth / 2;
      final y = size.height / 2;
      canvas.drawLine(
        Offset(x, y - heights[i] / 2),
        Offset(x, y + heights[i] / 2),
        paint..strokeWidth = barWidth * 0.75,
      );
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
