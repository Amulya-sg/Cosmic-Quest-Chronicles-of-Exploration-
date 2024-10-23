import 'package:flutter/material.dart';
import 'package:space_app/description.dart';
import 'package:space_app/quiz.dart';
import 'dart:ui';

class StarPainter extends CustomPainter {
  final Function(int) onStarTap;
  final int clickedStarIndex;

  StarPainter(this.onStarTap, this.clickedStarIndex);

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()..style = PaintingStyle.fill;

    final List<Offset> starPositions = [
      Offset(100, 100),
      Offset(200, 150),
      Offset(300, 200),
      Offset(150, 300),
    ];

    // Define colors for the stars
    final List<Color> starColors = [
      Color.fromRGBO(200, 200, 200, 1), // Gray for the first star
      Colors.orange, // Orange for the second star
      Colors.blue, // Blue for the third star
      Colors.red, // Red for the fourth star
    ];

    for (int i = 0; i < starPositions.length; i++) {
      final Offset position = starPositions[i];
      double starSize = (clickedStarIndex == i) ? 12 : 8;

      // Apply the color based on the star index
      paint.color = starColors[i];

      // Draw the main star
      canvas.drawCircle(position, starSize, paint);

      // Draw a smaller glow effect around the star
      paint.color = paint.color.withOpacity(0.5);
      canvas.drawCircle(position, starSize + 4, paint); // Decreased glow size
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}

// Main Gameplay class
class Gameplay extends StatefulWidget {
  @override
  _GameplayState createState() => _GameplayState();
}

class _GameplayState extends State<Gameplay>
    with SingleTickerProviderStateMixin<Gameplay> {
  bool isCircleClicked = false;
  int clickedCircleIndex = -1;
  bool isMissionEnabled = false; // State for the checkbox

  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    // Animation controller for the pulsating effect
    _controller = AnimationController(
      duration: Duration(seconds: 1),
      vsync: this,
    )..repeat(reverse: true);
  }

  void handleStarTap(int index) {
    setState(() {
      clickedCircleIndex = index;
      isCircleClicked = true;
    });
  }

  void closeContainer() {
    setState(() {
      isCircleClicked = false;
      clickedCircleIndex = -1;
      isMissionEnabled = false; // Reset checkbox state on close
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background Image
          Container(
            width: MediaQuery.of(context).size.width,
            height: MediaQuery.of(context).size.height,
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage('images/vecky.jpg'), // Path to your image
                fit: BoxFit.cover,
              ),
            ),
          ),
          // CustomPaint for Stars
          GestureDetector(
            onTapUp: (details) {
              // Get the position of the tap
              final tapPosition = details.localPosition;

              // Define star positions
              final List<Offset> starPositions = [
                Offset(100, 100),
                Offset(200, 150),
                Offset(300, 200),
                Offset(150, 300),
              ];

              // Check if the tap was within the range of any star
              for (int i = 0; i < starPositions.length; i++) {
                if ((tapPosition - starPositions[i]).distance < 30) {
                  handleStarTap(i);
                  break;
                }
              }
            },
            child: Stack(
              children: [
                CustomPaint(
                  size: Size(MediaQuery.of(context).size.width,
                      MediaQuery.of(context).size.height),
                  painter: StarPainter(handleStarTap, clickedCircleIndex),
                ),
                // Position texts below the stars
                Positioned(
                  left: 80,
                  top: 120, // Position text for the first star
                  child: Text(
                    "Kepler-452b",
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                ),
                Positioned(
                  left: 180,
                  top: 170, // Position text for the second star
                  child: Text(
                    "51-Pegasib",
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                ),
                Positioned(
                  left: 280,
                  top: 220, // Position text for the third star
                  child: Text(
                    "HD 189733b",
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                ),
                Positioned(
                  left: 130,
                  top: 320, // Position text for the fourth star
                  child: Text(
                    "GJ 1132b",
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                ),
              ],
            ),
          ),
          // Transparent Container
          if (isCircleClicked)
            Positioned(
              right: 10,
              top: MediaQuery.of(context).size.height * 0.2,
              child: transparentContainer(clickedCircleIndex),
            ),
        ],
      ),
    );
  }

  // Transparent Container that pops up with Template, Checkbox, and Button
  Widget transparentContainer(int index) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 5, sigmaY: 5), // Optional blur effect
        child: Container(
          width: 250,
          padding: EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.4),
            border: Border.all(
                color: Colors.transparent, width: 2), // Transparent border
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Button for description
              Container(
                margin: EdgeInsets.only(bottom: 10),
                child: customButton(
                  text: "Story",
                  onPressed: () {
                    Navigator.pushNamed(
                      context,
                      'planet_story', // Use the new route name
                      arguments: index, // Pass the selected index
                    );
                  },
                ),
              ),
              // Button for another action
              Container(
                margin: EdgeInsets.only(bottom: 20),
                child: customButton(
                  text: "Description",
                  onPressed: () {
                    Navigator.pushNamed(
                      context,
                      'description', // Use the new route name
                      arguments: index, // Pass the selected index
                    );
                  },
                ),
              ),
              Row(
                children: [
                  Checkbox(
                    value: isMissionEnabled, // Bind checkbox to state
                    onChanged: (bool? value) {
                      setState(() {
                        isMissionEnabled =
                            value ?? false; // Update checkbox state
                      });
                    },
                    activeColor: Colors.green,
                  ),
                  Text(
                    "Completed reading",
                    style: TextStyle(color: Colors.white),
                  ),
                ],
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: isMissionEnabled
                    ? () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => QuizPage(
                                clickedCircleIndex:
                                    clickedCircleIndex), // Pass the index
                          ),
                        );
                      }
                    : null,
                style: ButtonStyle(
                  backgroundColor: MaterialStateProperty.all(
                    isMissionEnabled
                        ? Colors.white
                        : Colors.grey, // Change color based on checkbox state
                  ),
                ),
                child: Text(
                  "Proceed",
                  style: TextStyle(color: Colors.black),
                ),
              ),
              SizedBox(height: 10),
              ElevatedButton(
                onPressed: closeContainer,
                style: ButtonStyle(
                  backgroundColor: MaterialStateProperty.all(Colors.white),
                ),
                child: Text(
                  "Close",
                  style: TextStyle(color: Colors.black),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // Custom button widget with arrow
  Widget customButton({required String text, required VoidCallback onPressed}) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.transparent, width: 2),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Expanded(
            child: TextButton(
              onPressed: onPressed,
              child: Text(
                text,
                style:
                    TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
          ),
          CustomPaint(
            size: Size(24, 24),
            painter: TrianglePainter(),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}

// Custom painter for triangle arrow
class TrianglePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.white.withOpacity(0.8);
    final path = Path()
      ..moveTo(0, 0)
      ..lineTo(size.width, size.height / 2)
      ..lineTo(0, size.height)
      ..close();

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
