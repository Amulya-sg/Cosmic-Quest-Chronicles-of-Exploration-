class Question {
  final String question;
  final List<String> options;
  final String correctAnswer;

  Question({
    required this.question,
    required this.options,
    required this.correctAnswer,
  });
}

// List of questions
List<Question> spaceQuizQuestions = [
  Question(
    question: "Which planet is known as the 'Red Planet'?",
    options: ["Mars", "Jupiter", "Venus", "Saturn"],
    correctAnswer: "Mars",
  ),
  Question(
    question: "What is the name of the galaxy that contains our Solar System?",
    options: [
      "Andromeda Galaxy",
      "Whirlpool Galaxy",
      "Milky Way Galaxy",
      "Sombrero Galaxy"
    ],
    correctAnswer: "Milky Way Galaxy",
  ),
  Question(
    question: "What force keeps planets in orbit around the Sun?",
    options: ["Magnetism", "Gravity", "Inertia", "Friction"],
    correctAnswer: "Gravity",
  ),
  Question(
    question: "How many moons does Earth have?",
    options: ["1", "2", "4", "5"],
    correctAnswer: "1",
  ),
  Question(
    question: "Which is the hottest planet in our solar system?",
    options: ["Mercury", "Venus", "Earth", "Mars"],
    correctAnswer: "Venus",
  ),
  Question(
    question:
        "What is the name of the first artificial satellite sent into space?",
    options: ["Apollo 11", "Sputnik 1", "Voyager 1", "Hubble"],
    correctAnswer: "Sputnik 1",
  ),
  Question(
    question: "What is the closest star to Earth?",
    options: ["Proxima Centauri", "Sirius", "Alpha Centauri", "The Sun"],
    correctAnswer: "The Sun",
  ),
  Question(
    question: "Which planet has the most moons?",
    options: ["Earth", "Jupiter", "Mars", "Saturn"],
    correctAnswer: "Jupiter",
  ),
  Question(
    question: "What is the term for a star that has exploded?",
    options: ["Nebula", "Black Hole", "Supernova", "Quasar"],
    correctAnswer: "Supernova",
  ),
  Question(
    question: "Which planet is known for its prominent ring system?",
    options: ["Jupiter", "Neptune", "Saturn", "Uranus"],
    correctAnswer: "Saturn",
  ),
  Question(
    question: "What is the smallest planet in our solar system?",
    options: ["Mercury", "Mars", "Venus", "Pluto"],
    correctAnswer: "Mercury",
  ),
  Question(
    question: "What is the name of the largest moon of Saturn?",
    options: ["Europa", "Titan", "Io", "Ganymede"],
    correctAnswer: "Titan",
  ),
  Question(
    question: "How many planets are there in our solar system?",
    options: ["7", "8", "9", "10"],
    correctAnswer: "8",
  ),
  Question(
    question: "Which planet is known for having a Great Red Spot?",
    options: ["Mars", "Jupiter", "Saturn", "Neptune"],
    correctAnswer: "Jupiter",
  ),
  Question(
    question: "What is the most abundant gas in Earth's atmosphere?",
    options: ["Oxygen", "Carbon Dioxide", "Hydrogen", "Nitrogen"],
    correctAnswer: "Nitrogen",
  ),
  Question(
    question: "Which planet is furthest from the Sun?",
    options: ["Uranus", "Neptune", "Pluto", "Saturn"],
    correctAnswer: "Neptune",
  ),
  Question(
    question: "What shape is the Milky Way galaxy?",
    options: ["Elliptical", "Spiral", "Irregular", "Circular"],
    correctAnswer: "Spiral",
  ),
  Question(
    question: "Which spacecraft was the first to land humans on the Moon?",
    options: ["Vostok 1", "Voyager 1", "Apollo 11", "Gemini 12"],
    correctAnswer: "Apollo 11",
  ),
  Question(
    question:
        "What is the term for a rock that enters Earth's atmosphere and burns up?",
    options: ["Comet", "Asteroid", "Meteor", "Planetoid"],
    correctAnswer: "Meteor",
  ),
  Question(
    question:
        "Which planet is called Earth's twin due to its similar size and composition?",
    options: ["Mars", "Venus", "Neptune", "Mercury"],
    correctAnswer: "Venus",
  ),
  Question(
    question: "How long does light from the Sun take to reach Earth?",
    options: ["8 minutes", "1 hour", "24 seconds", "8 seconds"],
    correctAnswer: "8 minutes",
  ),
  Question(
    question: "What causes a solar eclipse?",
    options: [
      "The Moon passing behind Earth",
      "The Sun moving away",
      "The Moon passing between Earth and the Sun",
      "The Earth rotating faster"
    ],
    correctAnswer: "The Moon passing between Earth and the Sun",
  ),
  Question(
    question: "Which planet spins on its side, unlike other planets?",
    options: ["Uranus", "Venus", "Mars", "Jupiter"],
    correctAnswer: "Uranus",
  ),
  Question(
    question: "What are Saturn's rings primarily made of?",
    options: ["Gas", "Ice and rock", "Dust", "Magnetic fields"],
    correctAnswer: "Ice and rock",
  ),
  Question(
    question:
        "Which planet has the fastest rotation, completing a day in about 10 hours?",
    options: ["Mars", "Earth", "Saturn", "Jupiter"],
    correctAnswer: "Jupiter",
  ),
  Question(
    question: "What is the brightest star in the night sky?",
    options: ["Betelgeuse", "Sirius", "Alpha Centauri", "Vega"],
    correctAnswer: "Sirius",
  ),
  Question(
    question: "Which planet has a year that lasts nearly 165 Earth years?",
    options: ["Neptune", "Saturn", "Uranus", "Pluto"],
    correctAnswer: "Neptune",
  ),
  Question(
    question: "What is the name of the first human to travel into space?",
    options: ["Neil Armstrong", "Yuri Gagarin", "Buzz Aldrin", "John Glenn"],
    correctAnswer: "Yuri Gagarin",
  ),
  Question(
    question: "What is a black hole?",
    options: [
      "A giant star",
      "A dense region of space with strong gravity",
      "A collapsing planet",
      "A distant galaxy"
    ],
    correctAnswer: "A dense region of space with strong gravity",
  ),
  Question(
    question: "What phenomenon is responsible for the Northern Lights?",
    options: [
      "Volcanic activity",
      "Solar wind interacting with Earth's atmosphere",
      "Meteor showers",
      "The rotation of Earth"
    ],
    correctAnswer: "Solar wind interacting with Earth's atmosphere",
  ),
];
