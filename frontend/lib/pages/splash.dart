// import 'package:flutter/material.dart';

// void main() => runApp(const MyApp());

// class MyApp extends StatelessWidget {
//   const MyApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       title: 'Generate Music',
//       theme: ThemeData(
//         fontFamily: 'Montserrat', // Use a modern sans-serif font as in the PDF
//         primaryColor: const Color(0xFFEFB7C6), // Pink pastel
//         scaffoldBackgroundColor: const Color(0xFFF8F8FF), // Soft background
//       ),
//       home: const SplashPage(),
//       debugShowCheckedModeBanner: false,
//     );
//   }
// }

// class SplashPage extends StatelessWidget {
//   const SplashPage({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Stack(
//         children: [
//           // Background with pastel blobs
//           CustomPaint(
//             size: MediaQuery.of(context).size,
//             painter: PastelBackgroundPainter(),
//           ),
//           Center(
//             child: Container(
//               width: 140,
//               height: 140,
//               decoration: BoxDecoration(
//                 color: const Color(0xFFEFB7C6), // Pink pastel
//                 borderRadius: BorderRadius.circular(20),
//                 border: Border.all(
//                   color: Colors.white,
//                   width: 4,
//                 ),
//                 boxShadow: [
//                   BoxShadow(
//                     color: Colors.black.withOpacity(0.07),
//                     blurRadius: 10,
//                     offset: const Offset(0, 6),
//                   ),
//                 ],
//               ),
//               child: Center(
//                 child: Stack(
//                   alignment: Alignment.center,
//                   children: [
//                     // The "B" with waveform and pastel overlay
//                     Text(
//                       'B',
//                       style: TextStyle(
//                         fontSize: 80,
//                         fontWeight: FontWeight.bold,
//                         color: Colors.white,
//                         letterSpacing: 2,
//                       ),
//                     ),
//                     Positioned(
//                       left: 28,
//                       top: 30,
//                       child: Container(
//                         width: 25,
//                         height: 60,
//                         decoration: BoxDecoration(
//                           color: const Color(0xFFB6F2E5), // Mint pastel
//                           borderRadius: BorderRadius.circular(10),
//                         ),
//                       ),
//                     ),
//                     // Add a simple waveform as a decorative line
//                     Positioned(
//                       right: 22,
//                       top: 60,
//                       child: Container(
//                         width: 35,
//                         height: 8,
//                         decoration: BoxDecoration(
//                           color: const Color(0xFFB6F2E5), // Mint pastel
//                           borderRadius: BorderRadius.circular(4),
//                         ),
//                       ),
//                     ),
//                   ],
//                 ),
//               ),
//             ),
//           ),
//         ],
//       ),
//     );
//   }
// }

// // Painter for pastel blobs background
// class PastelBackgroundPainter extends CustomPainter {
//   @override
//   void paint(Canvas canvas, Size size) {
//     final paint = Paint();

//     // Mint blob (top left)
//     paint.color = const Color(0xFFB6F2E5);
//     canvas.drawOval(
//       Rect.fromLTWH(-60, -30, size.width * 0.7, size.height * 0.5),
//       paint,
//     );

//     // Lavender blob (bottom right)
//     paint.color = const Color(0xFFD9C6F7);
//     canvas.drawOval(
//       Rect.fromLTWH(size.width * 0.2, size.height * 0.3, size.width * 0.9, size.height * 0.7),
//       paint,
//     );

//     // White central highlight
//     paint.color = Colors.white.withOpacity(0.85);
//     canvas.drawOval(
//       Rect.fromLTWH(size.width * 0.15, size.height * 0.1, size.width * 0.7, size.height * 0.7),
//       paint,
//     );
//   }

//   @override
//   bool shouldRepaint(CustomPainter oldDelegate) => false;
// }
import 'library_screen.dart';
import 'package:flutter/material.dart';
import 'package:blobs/blobs.dart';
import 'dart:math';
import 'blob_animation_controller.dart';



// ...existing SplashScreen class...

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  late BlobAnimationController blobAnimationController;
  late List<double> randomOffsets;

  // Icon animation fields
  late AnimationController iconController;
  late Animation<Offset> iconOffsetAnimation;

  // Text fade animation fields
  late AnimationController textController;
  late Animation<double> textFadeAnimation;

  final List<Color> blobColors = [
    Color(0xffeb79a2).withOpacity(0.3),
    Colors.purple.withOpacity(0.3),
    Color(0xFFEB79A2).withOpacity(0.3),
    Color(0xFFF7B267).withOpacity(0.3),
    Color(0xFFB2DBBF).withOpacity(0.3),
    const Color.fromARGB(255, 236, 12, 210).withOpacity(0.3),
    Colors.purple.withOpacity(0.3),
    Color(0xFF70C1B3).withOpacity(0.3),
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      setState(() {});
    });
    blobAnimationController = BlobAnimationController(vsync: this)
      ..setDuration(const Duration(seconds: 4));
    blobAnimationController.start();

    final random = Random();
    randomOffsets =
        List.generate(blobColors.length, (_) => random.nextDouble() * 40 - 20);

    // Icon animation setup
    iconController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );
    iconOffsetAnimation = Tween<Offset>(
      begin: Offset.zero,
      end: const Offset(-0.3, 0), // Move slightly left
    ).animate(CurvedAnimation(
      parent: iconController,
      curve: Curves.easeInOut,
    ));

    // Text fade animation setup
    textController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    textFadeAnimation = CurvedAnimation(
      parent: textController,
      curve: Curves.easeIn,
    );

    // Start icon animation after 3 seconds, then fade in text
    Future.delayed(const Duration(seconds: 3), () async {
      if (mounted) {
        await iconController.forward();
        if (mounted) {
          await textController.forward();
          // Navigate 3 seconds after "BEAT IT" appears
          Future.delayed(const Duration(seconds: 3), () {
            if (mounted) {
              Navigator.of(context).pushReplacement(
                MaterialPageRoute(builder: (_) => SplashSignInPage()),
              );
            }
          });
        }
      }
    });
  }

  @override
  void dispose() {
    blobAnimationController.dispose();
    iconController.dispose();
    textController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final double screenHeight = MediaQuery.of(context).size.height;
    final double screenWidth = MediaQuery.of(context).size.width;
    final double blobBaseSize =
        (screenHeight > screenWidth ? screenHeight : screenWidth);

    final List<double> blobSizes = [
      blobBaseSize * 1.2,
      blobBaseSize * 1.15,
      blobBaseSize * 1.1,
      blobBaseSize * 1.25,
      blobBaseSize * 1.18,
      blobBaseSize * 1.22,
      blobBaseSize * 1.13,
      blobBaseSize * 1.3,
    ];

    final List<double> verticalOffsets = [
      -blobSizes[0],
      -blobSizes[1],
      -blobSizes[2],
      -blobSizes[3],
      -blobSizes[4],
      -blobSizes[5],
      -blobSizes[6],
      -blobSizes[7],
    ];

    final List<double> horizontalOffsets = [
      screenWidth * 0.05,
      screenWidth * 0.25,
      screenWidth * 0.15,
      screenWidth * 0.45,
      screenWidth * 0.0,
      screenWidth * 0.6,
      screenWidth * 0.1,
      screenWidth * 0.5,
    ];

    final double stagger = 0.10;

    return Scaffold(
      backgroundColor: const Color(0xFFF4F6F8),
      body: Stack(
        children: [
          AnimatedBuilder(
            animation: blobAnimationController.controller,
            builder: (context, child) {
              return Stack(
                children: List.generate(blobColors.length, (index) {
                  double start = index * stagger;
                  double t =
                      ((blobAnimationController.controller.value - start) /
                              (1 - start))
                          .clamp(0.0, 1.0);

                  final double top = lerpDouble(
                    screenHeight * 1.2,
                    verticalOffsets[index],
                    t,
                  )!;

                  final double left =
                      horizontalOffsets[index % horizontalOffsets.length] +
                          randomOffsets[index] -
                          blobSizes[index] * 0.5;

                  return Positioned(
                    top: top,
                    left: left,
                    child: Blob.animatedRandom(
                      size: blobSizes[index],
                      styles: BlobStyles(
                        color: blobColors[index],
                      ),
                      minGrowth: 6,
                      edgesCount: 7 + index,
                      loop: false,
                    ),
                  );
                }),
              );
            },
          ),
          // The icon/logo always stays centered and on top, but now animated
          Center(
            child: SlideTransition(
              position: iconOffsetAnimation,
              child: SizedBox(
                width: 150,
                height: 150,
                child: Image.asset(
                  'assets/images/Frame 1.png',
                  width: 120,
                  height: 120,
                  fit: BoxFit.contain,
                ),
              ),
            ),
          ),
          // Vertical "BEAT IT" text fade in after icon moves
          Align(
            alignment: Alignment.centerRight,
            child: FadeTransition(
              opacity: textFadeAnimation,
              child: Padding(
                padding: const EdgeInsets.only(right: 70.0),
                child: RotatedBox(
                  quarterTurns: 1,
                  child: Text(
                    "BEAT IT",
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 70,
                      fontWeight: FontWeight.w900,
                      color: Color.fromARGB(255, 221, 242, 245),
                      letterSpacing: 1.5,
                      shadows: [
                        Shadow(
                          blurRadius: 12,
                          color: Color(0xffeb79a2),
                          offset: Offset(0, 0),
                        )
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// Helper for lerpDouble (since it's not imported by default)
double? lerpDouble(num a, num b, double t) {
  return a + (b - a) * t;
}