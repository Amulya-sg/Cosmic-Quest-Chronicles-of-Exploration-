import 'package:flutter/material.dart';
import 'myplanet.dart'; // Import your widget

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Planet Prediction',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: Scaffold(
        appBar: AppBar(
          title: Text('Planet Type Prediction'),
        ),
        body: PlanetPredictionWidget(
            backendUrl: 'http://192.168.0.145:8000'), // Pass your backend URL
      ),
    );
  }
}
