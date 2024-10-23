// Filename: planet_prediction.dart

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

// Models for Planet Type Prediction
class PlanetTypePrediction {
  final String planet_type;

  PlanetTypePrediction({required this.planet_type});

  factory PlanetTypePrediction.fromJson(Map<String, dynamic> json) {
    return PlanetTypePrediction(
      planet_type: json['planet_type'],
    );
  }
}

class PlanetPredictionWidget extends StatefulWidget {
  final String backendUrl;

  PlanetPredictionWidget({required this.backendUrl});

  @override
  _PlanetPredictionWidgetState createState() => _PlanetPredictionWidgetState();
}

class _PlanetPredictionWidgetState extends State<PlanetPredictionWidget> {
  final _formKey = GlobalKey<FormState>();
  final TextEditingController _massMultiplierController =
      TextEditingController();
  final TextEditingController _massWrtController = TextEditingController();
  final TextEditingController _radiusMultiplierController =
      TextEditingController();
  final TextEditingController _radiusWrtController = TextEditingController();
  final TextEditingController _distanceController = TextEditingController();
  final TextEditingController _stellarMagnitudeController =
      TextEditingController();
  final TextEditingController _orbitalRadiusController =
      TextEditingController();
  final TextEditingController _discoveryYearController =
      TextEditingController();

  PlanetTypePrediction? _prediction;
  bool _isLoadingPrediction = false;

  // Function to Predict Planet Type
  Future<void> _predictPlanetType() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoadingPrediction = true;
      _prediction = null;
    });

    try {
      var url = Uri.parse('${widget.backendUrl}/predict_planet_type');
      var response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'mass_multiplier': double.parse(_massMultiplierController.text),
          'mass_wrt': _massWrtController.text.trim(),
          'radius_multiplier': double.parse(_radiusMultiplierController.text),
          'radius_wrt': _radiusWrtController.text.trim(),
          'distance': double.parse(_distanceController.text),
          'stellar_magnitude': double.parse(_stellarMagnitudeController.text),
          'orbital_radius': double.parse(_orbitalRadiusController.text),
          'discovery_year': int.parse(_discoveryYearController.text),
        }),
      );

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        var prediction = PlanetTypePrediction.fromJson(data);
        setState(() {
          _prediction = prediction;
        });
      } else {
        var error = jsonDecode(response.body);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content:
                  Text(error['detail'] ?? 'Error predicting planet type.')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to connect to the server.')),
      );
    } finally {
      setState(() {
        _isLoadingPrediction = false;
      });
    }
  }

  // Widget to Display Prediction Result
  Widget _buildPredictionResult() {
    if (_prediction == null) return Container();
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Text(
        'Predicted Planet Type: ${_prediction!.planet_type}',
        style: TextStyle(
            fontSize: 18.0, fontWeight: FontWeight.bold, color: Colors.white),
      ),
    );
  }

  // Widget to Build Prediction Form
  Widget _buildPredictionForm() {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          Text(
            'Predict Planet Type',
            style: TextStyle(
                fontSize: 20.0,
                fontWeight: FontWeight.bold,
                color: Colors.white),
          ),
          SizedBox(height: 16.0),
          // Mass Multiplier
          TextFormField(
            controller: _massMultiplierController,
            decoration: InputDecoration(
              labelText: 'Mass Multiplier',
              labelStyle: TextStyle(
                color: Colors.grey, // Set the desired color for the label text
              ),
              border: OutlineInputBorder(),
            ),
            style: TextStyle(
              color: Colors.white, // Text color in TextField
              fontWeight: FontWeight.bold, // Make text bold
            ),
            keyboardType: TextInputType.numberWithOptions(decimal: true),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter mass multiplier.';
              }
              if (double.tryParse(value) == null) {
                return 'Please enter a valid number.';
              }
              return null;
            },
          ),
          SizedBox(height: 16.0),
          // Mass Reference
          TextFormField(
            controller: _massWrtController,
            decoration: InputDecoration(
              labelText: 'Mass Reference (e.g., Earth, Jupiter)',
              labelStyle: TextStyle(
                color: Colors.grey, // Set the desired color for the label text
              ),
              border: OutlineInputBorder(),
            ),
            style: TextStyle(
              color: Colors.white, // Text color in TextField
              fontWeight: FontWeight.bold, // Make text bold
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter mass reference.';
              }
              return null;
            },
          ),
          SizedBox(height: 16.0),
          // Radius Multiplier
          TextFormField(
            controller: _radiusMultiplierController,
            decoration: InputDecoration(
              labelText: 'Radius Multiplier',
              labelStyle: TextStyle(
                color: Colors.grey, // Set the desired color for the label text
              ),
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.numberWithOptions(decimal: true),
            style: TextStyle(
              color: Colors.white, // Text color in TextField
              fontWeight: FontWeight.bold, // Make text bold
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter radius multiplier.';
              }
              if (double.tryParse(value) == null) {
                return 'Please enter a valid number.';
              }
              return null;
            },
          ),
          SizedBox(height: 16.0),
          // Radius Reference
          TextFormField(
            controller: _radiusWrtController,
            decoration: InputDecoration(
              labelText: 'Radius Reference (e.g., Earth, Jupiter)',
              labelStyle: TextStyle(
                color: Colors.grey, // Set the desired color for the label text
              ),
              border: OutlineInputBorder(),
            ),
            style: TextStyle(
              color: Colors.white, // Text color in TextField
              fontWeight: FontWeight.bold, // Make text bold
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter radius reference.';
              }
              return null;
            },
          ),
          SizedBox(height: 16.0),
          // Distance
          TextFormField(
            controller: _distanceController,
            decoration: InputDecoration(
              labelText: 'Distance',
              labelStyle: TextStyle(
                color: Colors.grey, // Set the desired color for the label text
              ),
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.numberWithOptions(decimal: true),
            style: TextStyle(
              color: Colors.white, // Text color in TextField
              fontWeight: FontWeight.bold, // Make text bold
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter distance.';
              }
              if (double.tryParse(value) == null) {
                return 'Please enter a valid number.';
              }
              return null;
            },
          ),
          SizedBox(height: 16.0),
          // Stellar Magnitude
          TextFormField(
            controller: _stellarMagnitudeController,
            decoration: InputDecoration(
              labelText: 'Stellar Magnitude',
              labelStyle: TextStyle(
                color: Colors.grey, // Set the desired color for the label text
              ),
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.numberWithOptions(decimal: true),
            style: TextStyle(
              color: Colors.white, // Text color in TextField
              fontWeight: FontWeight.bold, // Make text bold
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter stellar magnitude.';
              }
              if (double.tryParse(value) == null) {
                return 'Please enter a valid number.';
              }
              return null;
            },
          ),
          SizedBox(height: 16.0),
          // Orbital Radius
          TextFormField(
            controller: _orbitalRadiusController,
            decoration: InputDecoration(
              labelText: 'Orbital Radius',
              labelStyle: TextStyle(
                color: Colors.grey, // Set the desired color for the label text
              ),
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.numberWithOptions(decimal: true),
            style: TextStyle(
              color: Colors.white, // Text color in TextField
              fontWeight: FontWeight.bold, // Make text bold
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter orbital radius.';
              }
              if (double.tryParse(value) == null) {
                return 'Please enter a valid number.';
              }
              return null;
            },
          ),
          SizedBox(height: 16.0),
          // Discovery Year
          TextFormField(
            controller: _discoveryYearController,
            decoration: InputDecoration(
              labelText: 'Discovery Year',
              labelStyle: TextStyle(
                color: Colors.grey, // Set the desired color for the label text
              ),
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.number,
            style: TextStyle(
              color: Colors.white, // Text color in TextField
              fontWeight: FontWeight.bold, // Make text bold
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter discovery year.';
              }
              if (int.tryParse(value) == null) {
                return 'Please enter a valid year.';
              }
              return null;
            },
          ),
          SizedBox(height: 16.0),
          // Predict Button
          ElevatedButton(
            onPressed: _isLoadingPrediction ? null : _predictPlanetType,
            child: Text('Predict Planet Type'),
          ),
          SizedBox(height: 16.0),
          // Loading Indicator
          if (_isLoadingPrediction) CircularProgressIndicator(),
          // Display Prediction Result
          _buildPredictionResult(),
        ],
      ),
    );
  }

  @override
  void dispose() {
    // Dispose controllers to free resources
    _massMultiplierController.dispose();
    _massWrtController.dispose();
    _radiusMultiplierController.dispose();
    _radiusWrtController.dispose();
    _distanceController.dispose();
    _stellarMagnitudeController.dispose();
    _orbitalRadiusController.dispose();
    _discoveryYearController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background Image
          Container(
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage(
                    'images/cosmos.jpeg'), // Replace with your image path
                fit: BoxFit.cover,
              ),
            ),
          ),
          // Left Container
          Positioned(
            left: 20,
            top: 0,
            bottom: 0,
            width: MediaQuery.of(context).size.width * 0.5,
            child: SingleChildScrollView(
              child: Container(
                color: Colors.black54, // Semi-transparent black
                padding: const EdgeInsets.all(16.0),
                child: _buildPredictionForm(),
              ),
            ),
          ),
          // Right Container for Prediction Result
          Positioned(
            right: 20,
            top: MediaQuery.of(context).size.height * 0.3,
            width: MediaQuery.of(context).size.width * 0.4,
            child: Center(
              child: Container(
                padding: const EdgeInsets.all(20.0),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.5),
                  borderRadius: BorderRadius.circular(15),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black26,
                      blurRadius: 10.0,
                      offset: Offset(0, 5),
                    ),
                  ],
                ),
                child: _buildPredictionResult(),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
