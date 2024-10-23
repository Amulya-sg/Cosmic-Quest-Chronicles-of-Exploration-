import 'package:flutter/material.dart';
import 'myplanet.dart';

class MyPlanetPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background image
          Container(
            decoration: BoxDecoration(
              image: DecorationImage(
                image:
                    AssetImage('images/cosmos.jpeg'), // Replace with your image
                fit: BoxFit.cover,
              ),
            ),
          ),
          Center(
            child: Container(
              width: 600, // Set the width of the transparent container
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.7), // Semi-transparent black
                borderRadius: BorderRadius.circular(10),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.5),
                    blurRadius: 20,
                    spreadRadius: 7,
                  ),
                ],
              ),
              child: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Display the feature description text
                    Text(
                      'Enter your choices to create your own planet\n\n'
                      'Features and Their Descriptions\n'
                      'Distance from Earth (in light years)\n'
                      'Description: This measures how far the planet is from Earth, expressed in light years—the distance that light travels in one year.\n'
                      'Input Range: 1 to 5000 light years (Feel free to use larger distances for theoretical planets).\n\n'
                      'Stellar Magnitude\n'
                      'Description: Stellar magnitude indicates the brightness of the star from which the planet orbits. A lower magnitude value means a brighter star.\n'
                      'Input Range: -10 to +15 (Negative values are very bright stars, while positive values indicate dim stars).\n\n'
                      'Discovery Year\n'
                      'Description: The year when the exoplanet was discovered. This helps to contextualize the planet within the timeline of astronomical discoveries.\n'
                      'Input Range: 1990 to present (Adjust as needed for theoretical contexts).\n\n'
                      'Mass Relative to Earth or Jupiter\n'
                      'Description: This represents the mass of the planet compared to Earth or Jupiter. Mass is a crucial factor in determining a planet\'s potential for hosting life.\n'
                      'Input Options: Select either "Earth" or "Jupiter."\n'
                      'Input Range: 0.1 to 50 Earth masses or 0.1 to 20 Jupiter masses (Use appropriate scaling for theoretical models).\n\n'
                      'Radius Relative to Earth or Jupiter\n'
                      'Description: Similar to mass, the radius is measured relative to either Earth or Jupiter. This affects gravity and potential atmosphere retention.\n'
                      'Input Options: Select either "Earth" or "Jupiter."\n'
                      'Input Range: 0.1 to 10 Earth radii or 0.1 to 5 Jupiter radii.\n\n'
                      'Orbital Radius (in Astronomical Units - AU)\n'
                      'Description: The distance of the planet from its star, expressed in astronomical units (AU), where 1 AU is the average distance from the Earth to the Sun.\n'
                      'Input Range: 0.1 to 30 AU (This covers close-in planets to those far from their stars).',
                      style: TextStyle(
                        color: Colors.white, // Text color
                        fontSize: 16,
                      ),
                      textAlign: TextAlign.justify,
                    ),
                    const SizedBox(height: 20),
                    ElevatedButton(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => PlanetPredictionWidget(
                                  backendUrl: 'http://192.168.0.145:8000')),
                        );
                      },
                      child: Text('Create'),
                    ),
                  ],
                ),
              ),
            ),
          )
        ],
      ),
    );
  }
}
