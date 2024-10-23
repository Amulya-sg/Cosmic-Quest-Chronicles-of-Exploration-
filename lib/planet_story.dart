import 'package:flutter/material.dart';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'database_helper.dart'; // Import your DatabaseHelper

class ImageTextPopup1 extends StatefulWidget {
  final int selectedPlanetIndex; // Planet index to retrieve data

  ImageTextPopup1({required this.selectedPlanetIndex}); // Constructor

  @override
  _ImageTextPopup1State createState() => _ImageTextPopup1State();
}

class _ImageTextPopup1State extends State<ImageTextPopup1>
    with SingleTickerProviderStateMixin {
  late DatabaseHelper _dbHelper;
  Exoplanet? _planet; // Store the fetched planet details

  // List of images and corresponding texts with placeholders
  final List<Map<String, String>> _data = [
    {
      'image': 'images/exoplanet.jpeg',
      'text': 'The planet name is {planetName}.'
    },
    {
      'image': 'images/hoststar.jpeg',
      'text': 'In the distance, {hoststar} glimmers with potential.'
    },
    {
      'image': 'images/distance.jpeg',
      'text':
          'This is a new world, {planetName}, located {distance} light-years away.'
    },
    // Add more entries with placeholders...
  ];

  @override
  void initState() {
    super.initState();
    _dbHelper = DatabaseHelper(); // Initialize DatabaseHelper
    _fetchPlanetDetails(); // Fetch planet details when initializing
  }

  Future<void> _fetchPlanetDetails() async {
    final planets = await _dbHelper.getAllPlanets(); // Fetch all planets
    if (widget.selectedPlanetIndex >= 0 &&
        widget.selectedPlanetIndex < planets.length) {
      setState(() {
        _planet =
            planets[widget.selectedPlanetIndex]; // Store the fetched planet
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    // Ensure the planet data is fetched before building the UI
    if (_planet == null) {
      return Center(
          child:
              CircularProgressIndicator()); // Show loading indicator while fetching
    }

    return Scaffold(
      backgroundColor: Colors.black, // Background color
      body: PageView.builder(
        itemCount: _data.length,
        itemBuilder: (context, index) {
          return _buildPageContent(_data[index], index);
        },
      ),
    );
  }

  Widget _buildPageContent(Map<String, String> data, int index) {
    // Use the fetched planet data
    String text = data['text']!
        .replaceFirst('{planetName}', _planet!.name) // Replace {planetName}
        .replaceFirst(
            '{distance}', _planet!.distance.toString()) // Replace {distance}
        .replaceFirst('{hoststar}', _planet!.hostStar);
    return Stack(
      children: [
        Positioned.fill(
          child: Image.asset(
            data['image']!,
            fit: BoxFit.cover,
          ),
        ),
        Align(
          alignment: Alignment.bottomCenter,
          child: Container(
            margin: EdgeInsets.all(16.0),
            padding: EdgeInsets.all(16.0),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.7),
              borderRadius: BorderRadius.circular(20),
            ),
            child: DefaultTextStyle(
              style: TextStyle(
                  fontSize: 20.0,
                  color: Colors.cyanAccent,
                  fontFamily: 'RobotoMono',
                  fontWeight: FontWeight.bold),
              child: AnimatedTextKit(
                animatedTexts: [
                  TypewriterAnimatedText(
                    text,
                    speed: Duration(milliseconds: 100),
                  ),
                ],
                isRepeatingAnimation: false,
              ),
            ),
          ),
        ),
        // Add button on the last page
        if (index == _data.length - 1)
          Positioned(
            bottom: 20,
            right: 20,
            child: ElevatedButton(
              onPressed: () {
                Navigator.pop(context); // Navigate back or to another page
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueAccent,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: Text('Next'),
            ),
          ),
      ],
    );
  }
}
