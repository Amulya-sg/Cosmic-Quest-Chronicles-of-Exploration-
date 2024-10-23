import 'dart:async';
import 'package:flutter/material.dart';
import 'package:space_app/clue_list.dart';
import 'package:space_app/gameplay.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: QuizPage(clickedCircleIndex: 0), // Default value for the index
    );
  }
}

class Question {
  final String questionText;
  final List<String> options;
  final String correctAnswer;

  Question({
    required this.questionText,
    required this.options,
    required this.correctAnswer,
  });
}

class QuizPage extends StatefulWidget {
  final int clickedCircleIndex; // Accept clickedCircleIndex

  QuizPage({required this.clickedCircleIndex}); // Modify constructor

  @override
  _QuizPageState createState() => _QuizPageState();
}

class _QuizPageState extends State<QuizPage> {
  final List<Question> questions = [
    Question(
      questionText: '1. How far is Kepler-452b from Earth in light years?',
      options: [
        '500 light year',
        '1400 light years',
        '2000 light years',
        '3000 light years'
      ],
      correctAnswer: '1400 light years',
    ),
    Question(
      questionText: 'What is the primary host star of Kepler-452b?',
      options: ['Sirius', 'ALpha Centauri', 'Kepler-452', 'Betelgeuse'],
      correctAnswer: 'Kepler-452',
    ),
    Question(
      questionText: 'What is the largest planet in our solar system?',
      options: ['Earth', 'Mars', 'Jupiter', 'Saturn'],
      correctAnswer: 'Jupiter',
    ),
    Question(
      questionText:
          'Kepler-452b is considered to be in which zone, making it potentially habitable?',
      options: ['Asteroid belt', 'Habitable belt', 'Kuiper belt', 'Oort cloud'],
      correctAnswer: 'Habitable belt',
    ),
    Question(
      questionText:
          'Which of the following elements is NOT mentioned as a possible component of Kepler-452b\'s atmosphere?',
      options: ['CH4', 'O2', 'CO2', 'N2'],
      correctAnswer: 'CH4',
    ),
  ];

  int currentQuestionIndex = 0;
  int score = 0;
  late Timer timer;
  int remainingTime = 15; // 15 seconds timer

  @override
  void initState() {
    super.initState();
    startTimer();
  }

  void startTimer() {
    remainingTime = 15; // Reset timer for each question
    timer = Timer.periodic(Duration(seconds: 1), (Timer timer) {
      setState(() {
        if (remainingTime > 0) {
          remainingTime--;
        } else {
          // Time is up; move to the next question
          answerQuestion('');
        }
      });
    });
  }

  void answerQuestion(String selectedOption) {
    if (timer.isActive) {
      timer.cancel(); // Cancel the timer when answering
    }

    if (selectedOption == questions[currentQuestionIndex].correctAnswer) {
      score++;
    }

    if (currentQuestionIndex < questions.length - 1) {
      setState(() {
        currentQuestionIndex++;
      });
      startTimer(); // Start timer for the next question
    } else {
      // Show result
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ResultPage(
            score: score,
            totalQuestions: questions.length,
            clickedCircleIndex:
                widget.clickedCircleIndex, // Pass clickedCircleIndex
          ),
        ),
      );
    }
  }

  @override
  void dispose() {
    timer.cancel(); // Clean up the timer
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
                    'images/spacestation.jpeg'), // Add your image path
                fit: BoxFit.cover,
              ),
            ),
          ),
          Positioned(
            top: 0,
            right: 200,
            child: Container(
              width: 500,
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.transparent, // Slightly transparent background
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.7),
                    spreadRadius: 1,
                    blurRadius: 10,
                  ),
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min, // Fit to content
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    questions[currentQuestionIndex].questionText,
                    style: TextStyle(fontSize: 24, color: Colors.white),
                  ),
                  SizedBox(height: 20),
                  Text(
                    'Time Remaining: $remainingTime seconds',
                    style: TextStyle(fontSize: 20, color: Colors.red),
                  ),
                  SizedBox(height: 20),
                  ...questions[currentQuestionIndex].options.map((option) {
                    return ElevatedButton(
                      onPressed: () => answerQuestion(option),
                      child: Text(option),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.tealAccent, // Sci-fi color
                      ),
                    );
                  }).toList(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class ResultPage extends StatelessWidget {
  final int score;
  final int totalQuestions;
  final int clickedCircleIndex; // Accept clickedCircleIndex

  ResultPage({
    required this.score,
    required this.totalQuestions,
    required this.clickedCircleIndex, // Add clickedCircleIndex
  });

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
                    'images/spacestation.jpeg'), // Add your image path
                fit: BoxFit.cover,
              ),
            ),
          ),
          Center(
            child: Container(
              padding: EdgeInsets.all(32),
              decoration: BoxDecoration(
                color:
                    Colors.black.withOpacity(0.7), // Slightly transparent black
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.5),
                    spreadRadius: 1,
                    blurRadius: 10,
                  ),
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Your score: $score / $totalQuestions',
                    style: TextStyle(fontSize: 24, color: Colors.white),
                    textAlign: TextAlign.center,
                  ),
                  SizedBox(height: 20),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      // Bottom left button (Back)
                      ElevatedButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (context) =>
                                    Gameplay()), // Navigate back to Gameplay page
                          );
                        },
                        child: Text('Back'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          padding: EdgeInsets.symmetric(
                              horizontal: 20, vertical: 10),
                          textStyle: TextStyle(fontSize: 18),
                        ),
                      ),
                      // Bottom right button (Next) - Only enabled if score is 5/5
                      ElevatedButton(
                        onPressed: score == totalQuestions
                            ? () {
                                // Navigate to CluePage with the selected star index
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) => CluePage(
                                        planetIndex: clickedCircleIndex),
                                  ),
                                );
                              }
                            : null, // Disabled if score is not 5/5
                        child: Text('Next'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.greenAccent,
                          padding: EdgeInsets.symmetric(
                              horizontal: 20, vertical: 10),
                          textStyle: TextStyle(fontSize: 18),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
