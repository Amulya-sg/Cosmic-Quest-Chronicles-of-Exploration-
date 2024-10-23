import 'package:flutter/material.dart';
import 'package:space_app/database_helper.dart'; // Import the DatabaseHelper

class LoginPage extends StatefulWidget {
  @override
  _LoginPageState createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage>
    with SingleTickerProviderStateMixin {
  // Animation controller and animation variables
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  double _imageOpacity = 0.0;

  final TextEditingController _usernameController =
      TextEditingController(); // Controller for the username input (New users)
  final TextEditingController _existingUserController =
      TextEditingController(); // Controller for existing user login

  final DatabaseHelper dbHelper =
      DatabaseHelper(); // Instance of DatabaseHelper

  @override
  void initState() {
    super.initState();

    // Initialize the controller for the pop-up box animation
    _controller = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );

    // Define the scale animation for the pop-up box
    _scaleAnimation =
        CurvedAnimation(parent: _controller, curve: Curves.easeInOut);

    // Trigger the image fade-in and then the box animation
    _startAnimations();
  }

  void _startAnimations() async {
    // Fade in the image over a longer duration, e.g., 7 seconds
    await Future.delayed(const Duration(
        milliseconds: 500)); // Small delay before the fade starts
    setState(() {
      _imageOpacity = 1.0;
    });

    // Wait for the image fade-in to complete
    await Future.delayed(
        const Duration(seconds: 5)); // Adjust for slower fade-in

    // Start the pop-up box animation
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose(); // Dispose of the animation controller
    _usernameController.dispose(); // Dispose of the username controller
    _existingUserController
        .dispose(); // Dispose of the existing user controller
    super.dispose();
  }

  void _login() async {
    String username =
        _usernameController.text.trim(); // Get username from input
    if (username.isNotEmpty) {
      int userId = await dbHelper.addUser(username); // Add user to the database

      // Check if the points entry for the user exists
      int points = await dbHelper.getUserPoints(userId);
      if (points == 0) {
        // If no points exist, create a new points entry with 20 points
        await dbHelper.createPointsEntry(userId, initialPoints: 20);
      }

      // Navigate to the main page after successful login
      Navigator.pushReplacementNamed(
        context,
        'main_page',
        arguments: userId, // Pass userId as an argument
      );
    } else {
      // Show a warning if the username is empty
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please enter a username')),
      );
    }
  }

  // Function to log in existing users
  void _loginExistingUser() async {
    String username = _existingUserController.text.trim();
    if (username.isNotEmpty) {
      int? userId = await dbHelper.getUserByUsername(
          username); // Check if the user exists in the database
      if (userId != null) {
        // If user exists, proceed to the main page
        Navigator.pushReplacementNamed(
          context,
          'main_page',
          arguments: userId, // Pass userId as an argument
        );
      } else {
        // If user does not exist, show an error message
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('User not found')),
        );
      }
    } else {
      // Show a warning if the username is empty
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please enter a username')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Use Image.asset for the background image
          AnimatedOpacity(
            opacity: _imageOpacity, // Controls the opacity level
            duration: const Duration(
                seconds: 7), // Fade-in duration adjusted for slower effect
            child: Container(
              width: double.infinity,
              height: double.infinity,
              child: Image.asset(
                'images/loginn.jpeg',
                fit: BoxFit.cover,
              ),
            ),
          ),
          SingleChildScrollView(
            // Wrap with SingleChildScrollView for scrollability
            child: Center(
              child: ScaleTransition(
                scale: _scaleAnimation,
                child: Container(
                  width: 400, // Set the width of the container
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.transparent, // Make the container transparent
                    borderRadius: BorderRadius.circular(10),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.5),
                        blurRadius: 20,
                        spreadRadius: 7,
                      ),
                    ],
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // Section for new users
                      Text(
                        'NEW USER',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Colors.white, // Change text color to cyan
                        ),
                      ),
                      const SizedBox(height: 10),
                      TextField(
                        controller: _usernameController,
                        decoration: InputDecoration(
                          labelText: 'Enter new username',
                          border: OutlineInputBorder(),
                          enabledBorder: OutlineInputBorder(
                            borderSide:
                                BorderSide(color: Colors.cyan), // Border color
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderSide: BorderSide(
                                color: Colors.blue), // Focused border color
                          ),
                        ),
                        style: TextStyle(
                          color: Colors.white, // Text color in TextField
                          fontWeight: FontWeight.bold, // Make text bold
                        ),
                      ),
                      const SizedBox(height: 20),
                      ElevatedButton(
                        onPressed:
                            _login, // Call the login function for new users
                        child: Text('Register'),
                      ),
                      const SizedBox(height: 10),
                      // Section for existing users
                      Text(
                        'EXISTING USER',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Colors.white, // Change text color to cyan
                        ),
                      ),
                      const SizedBox(height: 20),
                      TextField(
                        controller:
                            _existingUserController, // Controller for existing user login
                        decoration: InputDecoration(
                          labelText: 'Enter your username',
                          border: OutlineInputBorder(),
                          enabledBorder: OutlineInputBorder(
                            borderSide:
                                BorderSide(color: Colors.cyan), // Border color
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderSide: BorderSide(
                                color: Colors.blue), // Focused border color
                          ),
                        ),
                        style: TextStyle(
                          color: Colors.white, // Text color in TextField
                          fontWeight: FontWeight.bold, // Make text bold
                        ),
                      ),
                      const SizedBox(height: 20),
                      ElevatedButton(
                        onPressed:
                            _loginExistingUser, // Call the login function for existing users
                        child: Text('Login'),
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
}
