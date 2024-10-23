import 'package:flutter/material.dart';
import 'dart:async';
import 'package:space_app/gameplay.dart';
import 'package:space_app/database_helper.dart';
import 'package:space_app/myplanetintro.dart';
import 'package:space_app/quiz_categores.dart'; // Import the DatabaseHelper

class MainPage extends StatefulWidget {
  final int userId; // Add userId as a parameter

  MainPage({required this.userId}); // Constructor to accept userId

  @override
  _MainPageState createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> with TickerProviderStateMixin {
  late AnimationController _backgroundFadeController;
  late AnimationController _cardScaleController;
  bool _backgroundVisible = false;
  bool _cardsVisible = false;
  String _username = ""; // Variable to store the username
  int _points = 0; // Variable to store the user's points

  final DatabaseHelper dbHelper =
      DatabaseHelper(); // Instance of DatabaseHelper

  @override
  void initState() {
    super.initState();

    // Background fade-in animation controller
    _backgroundFadeController = AnimationController(
      vsync: this,
      duration: Duration(seconds: 4),
    );

    // Card pop-up animation controller
    _cardScaleController = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: 900),
    );

    // Fetch username and points from the database
    _fetchUsername();
    _fetchUserPoints(); // Fetch points

    // Start the background fade-in after a small delay
    Timer(Duration(milliseconds: 200), () {
      setState(() {
        _backgroundVisible = true;
      });
      _backgroundFadeController.forward();
    });

    // Delay the cards pop-up animation a little after the background
    Timer(Duration(seconds: 2), () {
      setState(() {
        _cardsVisible = true;
      });
      _cardScaleController.forward();
    });
  }

  Future<void> _fetchUsername() async {
    final user =
        await dbHelper.getUser(widget.userId); // Get user from database
    setState(() {
      _username = user?['username'] ?? "User"; // Set username
    });
  }

  Future<void> _fetchUserPoints() async {
    int points =
        await dbHelper.getUserPoints(widget.userId); // Get points from database
    setState(() {
      _points = points; // Update points
    });
  }

  @override
  void dispose() {
    _backgroundFadeController.dispose();
    _cardScaleController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Background image with fade-in effect inside a Stack
          AnimatedOpacity(
            opacity: _backgroundVisible ? 1.0 : 0.0,
            duration: Duration(seconds: 2),
            curve: Curves.easeInOut,
            child: Image.asset(
              'images/cosmos.jpeg',
              fit: BoxFit.cover,
              width: double.infinity,
              height: double.infinity,
            ),
          ),
          // Display username in a container at the top left
          Positioned(
            top: 26, // Move it to the top
            left: 5, // Move it to the left
            child: Container(
              padding: EdgeInsets.symmetric(vertical: 10, horizontal: 15),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.only(
                  bottomRight: Radius.circular(10),
                ),
              ),
              child: Text(
                _username, // Display the username
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          Positioned(
            top: 0, // Move it to the top
            right: 0, // Move it to the right
            child: Container(
              padding: EdgeInsets.symmetric(vertical: 10, horizontal: 15),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.only(
                  bottomLeft: Radius.circular(10),
                ),
              ),
              child: Text(
                'Points: $_points', // Display the points
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          // Cards pop-up animation, lowered for better positioning
          Positioned(
            top: 60, // Adjust this value to move the cards lower
            left: 0,
            right: 0,
            child: Center(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      buildAnimatedFuturisticCard(
                        'images/gameplay.jpeg',
                        'GAMEPLAY',
                        () {
                          // Navigate to GameplayPage
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => Gameplay()),
                          );
                        },
                      ),
                      buildAnimatedFuturisticCard(
                        'images/quiz.jpeg',
                        'QUIZ',
                        () {
                          // Navigate to GameplayPage
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (context) => QuizHomePage()),
                          );
                        },
                      ),
                      buildAnimatedFuturisticCard(
                        'images/make_planet.jpeg',
                        'MY PLANET',
                        () {
                          // Navigate to GameplayPage
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (context) => MyPlanetPage()),
                          );
                        },
                      ),
                      buildAnimatedFuturisticCard(
                        'images/shop.jpeg',
                        'SHOP',
                        null,
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // The existing methods remain unchanged...
  Widget buildAnimatedFuturisticCard(
      String imagePath, String title, VoidCallback? onTap) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedScale(
        scale: _cardsVisible ? 1.0 : 0.0,
        duration: Duration(milliseconds: 700),
        curve: Curves.elasticOut,
        child: Container(
          width: 160,
          height: 300,
          margin: EdgeInsets.symmetric(horizontal: 10.0),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            gradient: LinearGradient(
              colors: [Colors.deepPurple, Colors.blue],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black54,
                blurRadius: 10.0,
                offset: Offset(2, 5),
              ),
            ],
          ),
          child: Stack(
            children: [
              Positioned.fill(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Image.asset(
                    imagePath,
                    fit: BoxFit.cover,
                  ),
                ),
              ),
              Positioned(
                bottom: 40,
                left: 10,
                right: 10,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    sciFiButton(title, Colors.cyanAccent),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget sciFiButton(String text, Color borderColor) {
    return Container(
      padding: EdgeInsets.symmetric(vertical: 8.0, horizontal: 12.0),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10.0),
        border: Border.all(
          color: borderColor,
          width: 2.0,
        ),
        boxShadow: [
          BoxShadow(
            color: borderColor.withOpacity(0.5),
            blurRadius: 10,
            spreadRadius: 2,
          ),
        ],
        gradient: LinearGradient(
          colors: [Colors.black54, Colors.black87],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Text(
        text,
        textAlign: TextAlign.center,
        style: TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.bold,
          letterSpacing: 1.2,
          shadows: [
            Shadow(
              blurRadius: 10.0,
              color: borderColor.withOpacity(0.8),
              offset: Offset(0, 0),
            ),
          ],
        ),
      ),
    );
  }
}
