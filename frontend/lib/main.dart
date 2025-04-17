import 'package:flutter/material.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Create Account',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        fontFamily: 'Montserrat',
        scaffoldBackgroundColor: const Color(0xFFE6F7FF), // pastel blue
        primaryColor: const Color(0xFFFF4CA0), // pink
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          contentPadding: const EdgeInsets.symmetric(vertical: 16, horizontal: 18),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(30),
            borderSide: const BorderSide(color: Color(0xFFFF4CA0), width: 1.5),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(30),
            borderSide: const BorderSide(color: Color(0xFFFF4CA0), width: 2),
          ),
          hintStyle: const TextStyle(color: Colors.grey),
        ),
      ),
      home: const CreateAccountPage(),
    );
  }
}

class CreateAccountPage extends StatelessWidget {
  const CreateAccountPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Top pink blob
          Positioned(
            top: -70,
            left: -60,
            child: Blob(
              width: 220,
              height: 180,
              color: const Color(0xFFFF4CA0),
              opacity: 0.9,
            ),
          ),
          // Top right small circle
          Positioned(
            top: 40,
            right: 36,
            child: Container(
              width: 24,
              height: 24,
              decoration: const BoxDecoration(
                color: Color(0xFFFF4CA0),
                shape: BoxShape.circle,
              ),
            ),
          ),
          // Bottom pink blob
          Positioned(
            bottom: -60,
            left: -40,
            child: Blob(
              width: 230,
              height: 120,
              color: const Color(0xFFFF4CA0),
              opacity: 0.85,
            ),
          ),
          // Main content
          Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 26),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const SizedBox(height: 40),
                  const Text(
                    'Create new\nAccount',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFFFF4CA0),
                    ),
                  ),
                  const SizedBox(height: 8),
                  GestureDetector(
                    onTap: () {
                      // TODO: Navigate to Login
                    },
                    child: const Text(
                      'Already Registered? Login',
                      style: TextStyle(
                        color: Color(0xFFFF4CA0),
                        fontWeight: FontWeight.w500,
                        fontSize: 15,
                        decoration: TextDecoration.underline,
                      ),
                    ),
                  ),
                  const SizedBox(height: 32),
                  // Name Field
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Please enter your name:',
                      style: TextStyle(
                        color: Colors.grey[700],
                        fontWeight: FontWeight.w500,
                        fontSize: 14,
                      ),
                    ),
                  ),
                  const SizedBox(height: 6),
                  TextFormField(
                    initialValue: 'Aaron Loeb',
                    style: const TextStyle(fontSize: 16),
                    decoration: const InputDecoration(
                      hintText: 'Your Name',
                    ),
                  ),
                  const SizedBox(height: 18),
                  // Email Field
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Please enter Email:',
                      style: TextStyle(
                        color: Colors.grey[700],
                        fontWeight: FontWeight.w500,
                        fontSize: 14,
                      ),
                    ),
                  ),
                  const SizedBox(height: 6),
                  TextFormField(
                    initialValue: 'hello@reallygreatsite.com',
                    style: const TextStyle(fontSize: 16),
                    decoration: const InputDecoration(
                      hintText: 'Email Address',
                    ),
                  ),
                  const SizedBox(height: 18),
                  // Password Field
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Please enter Password:',
                      style: TextStyle(
                        color: Colors.grey[700],
                        fontWeight: FontWeight.w500,
                        fontSize: 14,
                      ),
                    ),
                  ),
                  const SizedBox(height: 6),
                  TextFormField(
                    initialValue: '*********',
                    obscureText: true,
                    style: const TextStyle(fontSize: 16),
                    decoration: const InputDecoration(
                      hintText: 'Password',
                    ),
                  ),
                  const SizedBox(height: 30),
                  // Sign Up Button
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFFFF4CA0),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(30),
                        ),
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        textStyle: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 18,
                        ),
                      ),
                      onPressed: () {},
                      child: const Text('SIGN UP'),
                    ),
                  ),
                  const SizedBox(height: 40),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// Simple blob shape using ClipPath and CustomPainter
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
