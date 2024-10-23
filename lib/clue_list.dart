import 'package:flutter/material.dart';
import 'package:space_app/gameplay.dart';

List<Map<String, String>> planets = [
  {
    "planet": "Kepler-452b",
    "message":
        "I’ve been searching for a way to keep things cool around here. It’s not too hot, not too cold, but I’m still figuring out how to manage the weather.",
    "character_inner_monologue":
        "This planet is more temperate compared to 51 Pegasi b. If the alien is still figuring things out, they might have some challenges to face, but at least it’s not an extreme environment."
  },
  {
    "planet": "51 Pegasi b",
    "message":
        "You wouldn’t believe it, I’ve been sweating like crazy these days. It’s almost unbearable, feels like the sun’s sitting right next to me. I hope things cool down soon, or I might just roast!",
    "character_inner_monologue":
        "Sweating like crazy… feels like the sun is next to them. This planet is scorching hot. 51 Pegasi b is practically touching its star, no one could last here for long. If the alien's friend is complaining about heat but hoping for things to cool down, it means they're definitely not here. This place is way too extreme for that. I feel like we need to move on."
  },
  // Add other planets here...
];

class CluePage extends StatelessWidget {
  final int planetIndex;

  CluePage({required this.planetIndex});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // PageView for the two clues
          PageView.builder(
            itemCount:
                2, // Each planet has two pages (message and inner monologue)
            itemBuilder: (context, index) {
              // Determine the content based on index
              String content;
              String imagePath;

              if (index == 0) {
                content = planets[planetIndex]['message']!;
                imagePath =
                    'images/backimage_0.jpeg'; // Add your image path for message
              } else {
                content = planets[planetIndex]['character_inner_monologue']!;
                imagePath =
                    'images/backimage_1.jpeg'; // Add your image path for inner monologue
              }

              return Stack(
                children: [
                  // Background Image
                  Container(
                    decoration: BoxDecoration(
                      image: DecorationImage(
                        image:
                            AssetImage(imagePath), // Set the appropriate image
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                  // Bottom Popup Container
                  Positioned(
                    bottom: 0,
                    left: 0,
                    right: 0,
                    child: Container(
                      padding: EdgeInsets.all(16.0),
                      margin: EdgeInsets.all(16.0),
                      decoration: BoxDecoration(
                        color: Colors.black
                            .withOpacity(0.7), // Semi-transparent background
                        borderRadius: BorderRadius.circular(12),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.5),
                            spreadRadius: 1,
                            blurRadius: 10,
                          ),
                        ],
                      ),
                      child: Text(
                        content,
                        style: TextStyle(fontSize: 20, color: Colors.white),
                      ),
                    ),
                  ),
                  // Button to check next planet only on the second page
                  if (index == 1)
                    Positioned(
                      top: 16,
                      right: 16,
                      child: ElevatedButton(
                        onPressed: () {
                          // Logic to check next planet (you might want to navigate to the next clue or planet)
                          Navigator.push(
                              context,
                              MaterialPageRoute(
                                  builder: (context) =>
                                      Gameplay())); // Example: navigate back (change as needed)
                        },
                        child: Text('Check Next Planet'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white, // Background color
                          padding:
                              EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        ),
                      ),
                    ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }
}
