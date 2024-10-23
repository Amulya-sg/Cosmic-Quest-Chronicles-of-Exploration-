import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:space_app/database_helper.dart'; // Import DatabaseHelper
import 'package:space_app/intro_page.dart';
import 'package:space_app/login.dart';
import 'package:space_app/main_page.dart';
import 'package:space_app/first_storyline.dart';
import 'package:space_app/description.dart';
import 'package:space_app/planet_story.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeRight,
    DeviceOrientation.landscapeLeft,
  ]).then((_) async {
    // Initialize the database and insert sample planets
    DatabaseHelper dbHelper = DatabaseHelper();
    await dbHelper.database; // Ensure database is initialized
    await dbHelper.insertSamplePlanets(); // Insert sample data
    runApp(const MyApp());
  });
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: IntroPage(),
      onGenerateRoute: (RouteSettings settings) {
        switch (settings.name) {
          case 'login':
            return MaterialPageRoute(builder: (context) => LoginPage());
          case 'first_storyline':
            return MaterialPageRoute(builder: (context) => Storyline());

          case 'main_page':
            final args =
                settings.arguments as int; // Extract userId from arguments
            return MaterialPageRoute(
              builder: (context) =>
                  MainPage(userId: args), // Pass userId to MainPage
            );
          case 'description':
            final args = settings.arguments as int; // Extract the planet index
            return MaterialPageRoute(
              builder: (context) =>
                  DescriptionPage(planetIndex: args), // Pass the index
            );
          case 'planet_story':
            final args = settings.arguments as int; // Extract the planet index
            return MaterialPageRoute(
              builder: (context) => ImageTextPopup1(
                selectedPlanetIndex: args, // Pass the index to ImageTextPopup1
              ),
            );
          default:
            return MaterialPageRoute(builder: (context) => IntroPage());
        }
      },
    );
  }
}
