import 'package:flutter/material.dart';
import 'database_helper.dart'; // Import your DatabaseHelper

class DescriptionPage extends StatelessWidget {
  final int planetIndex; // Receive the planet index

  DescriptionPage({required this.planetIndex});

  final DatabaseHelper _dbHelper = DatabaseHelper();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black, // Set the background to black
      body: FutureBuilder<Exoplanet?>(
        future: _fetchPlanetDetails(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(
                child: Text('Error: ${snapshot.error}',
                    style: TextStyle(color: Colors.white)));
          } else if (snapshot.hasData) {
            final planet = snapshot.data!;
            return Stack(
              children: [
                Positioned.fill(
                  child: Image.asset(
                    'images/exoplanet1.jpeg', // Replace with your image path
                    fit: BoxFit.cover,
                    color: Colors.black.withOpacity(0.7),
                    colorBlendMode: BlendMode.darken,
                  ),
                ),
                Positioned(
                  left: 40, // Adjust horizontal positioning
                  right: 40, // Adjust horizontal positioning
                  bottom: 40, // Position from bottom
                  child: Container(
                    width: double.infinity, // Full-width container
                    padding: const EdgeInsets.all(16.0),
                    decoration: BoxDecoration(
                      color: Colors.black
                          .withOpacity(0.5), // Sci-fi transparent look
                      borderRadius:
                          BorderRadius.circular(20), // More rounded corners
                      border: Border.all(
                        color: Colors.purpleAccent, // Sci-fi border color
                        width: 2,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.blueAccent.withOpacity(0.6),
                          blurRadius: 20,
                          spreadRadius: 5,
                          offset: Offset(0, 10),
                        ),
                      ],
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Text(
                          'Name: ${planet.name}',
                          style: TextStyle(
                            fontFamily: 'RobotoMono', // Sci-fi styled font
                            fontSize: 24,
                            color: Colors.cyanAccent,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        SizedBox(height: 8),
                        Text(
                          'Distance: ${planet.distance}',
                          style: TextStyle(
                            fontFamily: 'RobotoMono',
                            fontSize: 18,
                            color: Colors.white,
                          ),
                        ),
                        SizedBox(height: 8),
                        Text(
                          'Host Star: ${planet.hostStar}',
                          style: TextStyle(
                            fontFamily: 'RobotoMono',
                            fontSize: 18,
                            color: Colors.white,
                          ),
                        ),
                        SizedBox(height: 8),
                        Text(
                          'Size: ${planet.size}',
                          style: TextStyle(
                            fontFamily: 'RobotoMono',
                            fontSize: 18,
                            color: Colors.white,
                          ),
                        ),
                        SizedBox(height: 8),
                        Text(
                          'Potential for Life: ${planet.potentialForLife}',
                          style: TextStyle(
                            fontFamily: 'RobotoMono',
                            fontSize: 18,
                            color: Colors.white,
                          ),
                        ),
                        SizedBox(height: 8),
                        Text(
                          'Notable Feature: ${planet.notableFeature}',
                          style: TextStyle(
                            fontFamily: 'RobotoMono',
                            fontSize: 18,
                            color: Colors.white,
                          ),
                        ),
                        SizedBox(height: 16),
                        ElevatedButton(
                          onPressed: () {
                            Navigator.pop(context); // Back button functionality
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.cyanAccent, // Button color
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10),
                            ),
                            padding: EdgeInsets.symmetric(
                                horizontal: 32, vertical: 12),
                          ),
                          child: Text(
                            'Back',
                            style: TextStyle(
                              fontFamily: 'RobotoMono',
                              fontSize: 18,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            );
          } else {
            return Center(
                child: Text('No planet found.',
                    style: TextStyle(color: Colors.white)));
          }
        },
      ),
    );
  }

  Future<Exoplanet?> _fetchPlanetDetails() async {
    final planets = await _dbHelper.getAllPlanets();
    if (planetIndex >= 0 && planetIndex < planets.length) {
      return planets[planetIndex]; // Return the planet based on index
    }
    return null; // Return null if the index is out of bounds
  }
}
