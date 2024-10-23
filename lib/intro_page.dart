import 'package:flutter/material.dart';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:space_app/first_storyline.dart';

class IntroPage extends StatefulWidget {
  @override
  _IntroPageState createState() => _IntroPageState();
}

class _IntroPageState extends State<IntroPage> {
  bool _showSecondText = false; // To control when to display the second text
  bool _showNextButton = false; // To control when to display the Next button

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black87, // Dark retro background
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Center(
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 500),
              child: _showSecondText
                  ? AnimatedTextKit(
                      key: ValueKey(2),
                      animatedTexts: [
                        TypewriterAnimatedText(
                          'Get ready to embark on a journey!',
                          textStyle: const TextStyle(
                            fontFamily: 'Courier', // Retro typewriter font
                            color: Colors.cyanAccent,
                            fontSize: 24.0,
                            fontWeight: FontWeight.bold,
                          ),
                          speed: const Duration(milliseconds: 80),
                          cursor: '|', // Classic typewriter cursor
                        ),
                      ],
                      totalRepeatCount: 1,
                      onFinished: () {
                        setState(() {
                          _showNextButton =
                              true; // Show "Next" button after the second text
                        });
                      },
                    )
                  : AnimatedTextKit(
                      key: ValueKey(1),
                      animatedTexts: [
                        TypewriterAnimatedText(
                          'Welcome to our app',
                          textStyle: const TextStyle(
                            fontFamily: 'Courier', // Retro typewriter font
                            color: Colors.cyanAccent,
                            fontSize: 24.0,
                            fontWeight: FontWeight.bold,
                          ),
                          speed: const Duration(milliseconds: 80),
                          cursor: '|', // Classic typewriter cursor
                        ),
                      ],
                      totalRepeatCount: 1,
                      onFinished: () {
                        setState(() {
                          _showSecondText = true;
                        });
                      },
                    ),
            ),
          ),
          const SizedBox(
              height: 20), // Add some spacing between text and the button
          if (_showNextButton) // Show the Next button only after the second text is done
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor:
                    Color.fromARGB(255, 238, 237, 229), // Retro button color
                textStyle: const TextStyle(
                  fontFamily: 'Courier', // Matching retro font for the button
                  fontSize: 18,
                ),
              ),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => Storyline()),
                );
              },
              child: const Text(
                'Next',
                style:
                    TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
              ),
            ),
        ],
      ),
    );
  }
}
