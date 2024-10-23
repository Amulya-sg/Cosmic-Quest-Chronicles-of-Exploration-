import 'package:flutter/material.dart';
import 'list.dart'; // For rootBundle

class QuizHomePage extends StatefulWidget {
  @override
  _QuizHomePageState createState() => _QuizHomePageState();
}

class _QuizHomePageState extends State<QuizHomePage> {
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
                    'images/spacestation.jpeg'), // Replace with your background image path
                fit: BoxFit.cover,
              ),
            ),
          ),
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                buildAttemptContainer(context, 'Attempt 1', 1),
                buildAttemptContainer(context, 'Attempt 2', 2),
                buildAttemptContainer(context, 'Attempt 3', 3),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // Helper function to build attempt container
  Widget buildAttemptContainer(
      BuildContext context, String title, int attemptNumber) {
    return GestureDetector(
      onTap: () {
        // On tapping the container, navigate to the quiz page
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => QuizPage(attemptNumber: attemptNumber),
          ),
        );
      },
      child: Card(
        margin: EdgeInsets.symmetric(vertical: 10, horizontal: 20),
        color: Colors.black, // Black background for the card
        elevation: 8, // Elevation for the card effect
        child: Container(
          padding: EdgeInsets.all(20),
          child: Text(
            title,
            style: TextStyle(color: Colors.white, fontSize: 18),
          ),
        ),
      ),
    );
  }
}

// Quiz Page
class QuizPage extends StatefulWidget {
  final int attemptNumber;

  QuizPage({required this.attemptNumber});

  @override
  _QuizPageState createState() => _QuizPageState();
}

class _QuizPageState extends State<QuizPage> {
  late List<Question> selectedQuestions;
  int currentQuestionIndex = 0;
  int score = 0;

  @override
  void initState() {
    super.initState();
    // Shuffle and pick 5 random questions
    selectedQuestions = shuffleAndPick(spaceQuizQuestions, 5);
  }

  // Function to shuffle and pick random questions
  List<Question> shuffleAndPick(List<Question> questions, int count) {
    List<Question> shuffledQuestions = List<Question>.from(questions);
    shuffledQuestions.shuffle();
    return shuffledQuestions.take(count).toList();
  }

  void checkAnswer(String selectedAnswer) {
    if (selectedAnswer ==
        selectedQuestions[currentQuestionIndex].correctAnswer) {
      score++;
    }
    if (currentQuestionIndex < selectedQuestions.length - 1) {
      setState(() {
        currentQuestionIndex++;
      });
    } else {
      // Quiz is over, navigate to result page
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => ResultPage(
              score: score, totalQuestions: selectedQuestions.length),
        ),
      );
    }
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
                    'images/spacestation.jpeg'), // Replace with your background image path
                fit: BoxFit.cover,
              ),
            ),
          ),
          // Container for question and options
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Center(
              child: Container(
                padding: EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.black
                      .withOpacity(0.5), // Slightly transparent background
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
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Question ${currentQuestionIndex + 1}/${selectedQuestions.length}',
                      style: TextStyle(fontSize: 24, color: Colors.white),
                    ),
                    SizedBox(height: 20),
                    Text(
                      selectedQuestions[currentQuestionIndex].question,
                      style: TextStyle(fontSize: 18, color: Colors.white),
                    ),
                    SizedBox(height: 20),
                    ...selectedQuestions[currentQuestionIndex]
                        .options
                        .map((option) {
                      return ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.tealAccent, // Sci-fi color
                        ),
                        onPressed: () => checkAnswer(option),
                        child: Text(option),
                      );
                    }).toList(),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// Result Page
class ResultPage extends StatelessWidget {
  final int score;
  final int totalQuestions;

  ResultPage({required this.score, required this.totalQuestions});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Quiz Result'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'You scored $score out of $totalQuestions!',
              style: TextStyle(fontSize: 24),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => QuizHomePage()),
                );
              },
              child: Text('Return to Home'),
            ),
          ],
        ),
      ),
    );
  }
}
