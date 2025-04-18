import 'package:flutter/material.dart';

void main() {
  runApp(const MusicGeneratorApp());
}

class MusicGeneratorApp extends StatelessWidget {
  const MusicGeneratorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Music Generator',
      theme: ThemeData(
        primarySwatch: Colors.pink,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController textPromptController = TextEditingController();
  final TextEditingController styleReferencesController = TextEditingController();

  @override
  void dispose() {
    textPromptController.dispose();
    styleReferencesController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;
    final isPortrait = screenSize.height > screenSize.width;
    final double contentPadding = screenSize.width * 0.05;
    
    // App color scheme
    const Color appPink = Color(0xFFE91E63);
    const Color appDarkPink = Color(0xFFD81B60);
    const Color appBackground = Colors.black;
    const Color textColor = Colors.white;

    return Scaffold(
      backgroundColor: appBackground,
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: EdgeInsets.all(contentPadding),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // HEADER - GENERATE MUSIC
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Center(
                    child: Text(
                      'GENERATE MUSIC',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: appDarkPink,
                      ),
                    ),
                  ),
                ),
                
                const SizedBox(height: 12),
                
                // STEP 1
                Container(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  decoration: BoxDecoration(
                    color: appPink,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Center(
                    child: Text(
                      'Step 1',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
                
                const SizedBox(height: 16),
                
                // ENTER TEXT PROMPT
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(
                        vertical: 8, 
                        horizontal: 12
                      ),
                      child: const Text(
                        'ENTER TEXT PROMPT',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: textColor,
                        ),
                      ),
                    ),
                    Container(
                      decoration: BoxDecoration(
                        color: appPink.withOpacity(0.6),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: TextField(
                        controller: textPromptController,
                        style: const TextStyle(color: Colors.white),
                        maxLines: 3,
                        decoration: const InputDecoration(
                          hintText: 'Describe your dream composition in detail (e.g. Arabian pop with electronic elements)',
                          hintStyle: TextStyle(color: Colors.white70, fontSize: 12),
                          contentPadding: EdgeInsets.all(12),
                          border: InputBorder.none,
                        ),
                      ),
                    ),
                  ],
                ),
                
                const SizedBox(height: 20),
                
                // SETTINGS SECTION
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // VOCALS SETTINGS
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'VOCALS SETTINGS',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.bold,
                              color: textColor,
                            ),
                          ),
                          const SizedBox(height: 8),
                          _buildSettingButton('Vocal Type', screenSize),
                          _buildSettingButton('Vocal Pitch', screenSize),
                          _buildSettingButton('Vocal Range', screenSize),
                          _buildSettingButton('Upload Sample', screenSize),
                        ],
                      ),
                    ),
                    
                    SizedBox(width: isPortrait ? 12 : 24),
                    
                    // LYRICS SETTINGS
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'LYRICS SETTINGS',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.bold,
                              color: textColor,
                            ),
                          ),
                          const SizedBox(height: 8),
                          _buildSettingButton('Auto Generate', screenSize),
                          _buildSettingButton('Custom Lyrics', screenSize),
                          _buildSettingButton('Language', screenSize),
                          _buildSettingButton('Theme', screenSize),
                        ],
                      ),
                    ),
                  ],
                ),
                
                const SizedBox(height: 20),
                
                // STYLE REFERENCES
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: const [
                        Text(
                          'STYLE REFERENCES:',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            color: textColor,
                          ),
                        ),
                        SizedBox(width: 8),
                        Text(
                          'Sample Lyrics',
                          style: TextStyle(
                            fontSize: 12,
                            fontStyle: FontStyle.italic,
                            color: Colors.white70,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Container(
                      decoration: BoxDecoration(
                        color: appPink.withOpacity(0.6),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: TextField(
                        controller: styleReferencesController,
                        style: const TextStyle(color: Colors.white),
                        maxLines: 3,
                        decoration: const InputDecoration(
                          hintText: 'Add artist inspirations (e.g. "Like Queen meets Hans Zimmer") or paste song lyrics that you want to influence the style',
                          hintStyle: TextStyle(color: Colors.white70, fontSize: 12),
                          contentPadding: EdgeInsets.all(12),
                          border: InputBorder.none,
                        ),
                      ),
                    ),
                  ],
                ),
                
                const SizedBox(height: 20),
                
                // SAMPLE MUSIC TYPE
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      'Sample Music Type',
                      style: TextStyle(
                        fontSize: 14,
                        color: textColor,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Container(
                      width: 50,
                      height: 50,
                      decoration: BoxDecoration(
                        color: appPink,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.play_arrow,
                        color: Colors.white,
                        size: 30,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSettingButton(String label, Size screenSize) {
    final double buttonHeight = screenSize.height * 0.05;
    
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Container(
        height: buttonHeight,
        decoration: BoxDecoration(
          color: const Color(0xFFE91E63),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Center(
          child: Text(
            label,
            style: TextStyle(
              color: Colors.white,
              fontSize: screenSize.width < 360 ? 11 : 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ),
    );
  }
}
