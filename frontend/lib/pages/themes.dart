import 'package:flutter/material.dart';

class AppTheme {
  static ThemeData get lightTheme {
    return ThemeData(
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
    );
  }
}
