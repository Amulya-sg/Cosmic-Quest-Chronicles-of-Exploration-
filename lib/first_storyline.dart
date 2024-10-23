import 'package:flutter/material.dart';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:space_app/login.dart'; // For typing effect

class Storyline extends StatefulWidget {
  @override
  _StorylineState createState() => _StorylineState();
}

class _StorylineState extends State<Storyline> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: ImageTextPopup(),
    );
  }
}

class ImageTextPopup extends StatefulWidget {
  @override
  _ImageTextPopupState createState() => _ImageTextPopupState();
}

class _ImageTextPopupState extends State<ImageTextPopup>
    with SingleTickerProviderStateMixin {
  final PageController _pageController = PageController();
  int _currentPage = 0;
  bool _showText = false;
  bool _isFullTextDisplayed = false; // New variable for full text display

  late AnimationController _animationController;
  late Animation<double> _zoomAnimation;

  // List of images and corresponding texts
  final List<Map<String, String>> _data = [
    {
      'image': 'images/earth.jpeg',
      'text':
          'In the year 2045, amidst the rapid advancements in space exploration and technology, Dr. Alex and Dr. Orion work at the prestigious International Exoplanet Research Lab located in Greenwich, England. Here, they push the boundaries of human knowledge, delving into the mysteries of distant worlds beyond our solar system.'
    },
    {
      'image': 'images/friends.jpeg',
      'text':
          'Within the International Exoplanet Research Lab, Dr. Alex and Dr. Orion share a deep friendship, forged through years of collaboration. Late-night discussions blend theories of exoplanets with laughter and dreams, making them not just colleagues but a formidable team ready to change humanity\'s fate'
    },
    {
      'image': 'images/working.jpeg',
      'text':
          'Dr. Alex and Dr. Orion have been working tirelessly on groundbreaking research about exoplanets. Together, they believe they are on the verge of a discovery that could change humanity’s understanding of the universe.'
    },
    {
      'image': 'images/abduction.jpeg',
      'text':
          'Without warning, a mysterious spacecraft appears above their lab. The Galactic Dominion, an ancient alien organization, has been monitoring their work. Believing their discoveries to be a threat, the aliens abduct Dr. Orion.'
    },
    {
      'image': 'images/sad.jpeg',
      'text':
          'As Dr. Alex gazes at the stars through the lab\'s observatory, a heavy heart weighs him down. The uncertainty of their recent discoveries and the looming threat from the Galactic Dominion fill him with dread. He worries for Dr. Orion’s safety and feels the burden of their shared dreams slipping away'
    },
    {
      'image': 'images/device.jpeg',
      'text':
          'In the chaos, the aliens leave behind a strange electronic device. Upon examining it, Dr. Alex discovers encrypted coordinates to the exoplanet 51 Pegasi b, where the Galactic Dominion plans a meeting'
    },
    {
      'image': 'images/surprise.jpeg',
      'text':
          'The device reveals a cryptic message from the Galactic Dominion: \'All exoplanets fall under our domain. Your friend is being held on one of them. Any interference will be met with force.'
    },
    {
      'image': 'images/research.jpeg',
      'text':
          'Determined to save Dr. Orion, Dr. Alex immerses himself in research, poring over data and simulations of exoplanets'
    },
    {
      'image': 'images/determined.jpeg',
      'text':
          'Dr. Alex refits their spacecraft for long-distance space travel. With the coordinates to 51 Pegasi b locked in, it\'s time to launch and begin the journey to save Dr. Orion.'
    },
    {
      'image': 'images/leaving.jpeg',
      'text':
          'The journey brings Dr. Alex to Kepler 452b, a hot Jupiter exoplanet'
    },
  ];

  @override
  void initState() {
    super.initState();

    _animationController = AnimationController(
      vsync: this,
      duration: Duration(seconds: 3),
    );

    _zoomAnimation = Tween<double>(begin: 1.0, end: 1.1).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: Curves.easeInOut,
      ),
    );

    _animationController.forward();

    Future.delayed(Duration(milliseconds: 500), () {
      setState(() {
        _showText = true;
      });
    });
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return PageView.builder(
      controller: _pageController,
      itemCount: _data.length,
      onPageChanged: (int index) {
        _animationController.reset();
        _animationController.forward();
        setState(() {
          _currentPage = index;
          _showText = false;
          _isFullTextDisplayed = false; // Reset full text display
          Future.delayed(Duration(milliseconds: 500), () {
            setState(() {
              _showText = true;
            });
          });
        });
      },
      itemBuilder: (context, index) {
        return _buildPageContent(_data[index], index);
      },
    );
  }

  Widget _buildPageContent(Map<String, String> data, int index) {
    return Stack(
      children: [
        Positioned.fill(
          child: AnimatedBuilder(
            animation: _zoomAnimation,
            builder: (context, child) {
              return Transform.scale(
                scale: _zoomAnimation.value,
                child: Image.asset(
                  data['image']!,
                  fit: BoxFit.cover,
                ),
              );
            },
          ),
        ),
        Align(
          alignment: Alignment.bottomCenter,
          child: GestureDetector(
            onTap: () {
              // Toggle full text display on tap
              setState(() {
                _isFullTextDisplayed =
                    !_isFullTextDisplayed; // Toggle full text display
              });
            },
            child: AnimatedOpacity(
              opacity: _showText ? 1.0 : 0.0,
              duration: Duration(milliseconds: 500),
              child: Container(
                margin: EdgeInsets.only(bottom: 30),
                padding: EdgeInsets.all(16.0),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.blueAccent.withOpacity(0.5),
                      blurRadius: 20,
                      spreadRadius: 5,
                      offset: Offset(0, 10),
                    ),
                  ],
                  gradient: LinearGradient(
                    colors: [
                      Colors.blueAccent.withOpacity(0.3),
                      Colors.purpleAccent.withOpacity(0.4),
                      Colors.cyanAccent.withOpacity(0.4),
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  border: Border.all(
                    color: Colors.blueAccent.withOpacity(0.8),
                    width: 2,
                  ),
                ),
                width: double.infinity,
                child: _buildTypingText(data['text']!),
              ),
            ),
          ),
        ),
        // Add button on the last page
        if (_currentPage == _data.length - 1)
          Positioned(
            bottom: 20,
            right: 20,
            child: ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => LoginPage()),
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: Text('Next'),
            ),
          ),
        // Add skip button for all pages except the last one
        if (_currentPage < _data.length - 1)
          Positioned(
            top: 20,
            right: 20,
            child: ElevatedButton(
              onPressed: () {
                // Skip to the last page
                _pageController.jumpToPage(_data.length - 1);
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: Text('Skip'),
            ),
          ),
      ],
    );
  }

  // Function to create the typing effect text or show full text based on the state
  Widget _buildTypingText(String text) {
    return DefaultTextStyle(
      style: TextStyle(
          fontSize: 20.0,
          color: Colors.cyanAccent,
          fontFamily: 'RobotoMono',
          fontWeight: FontWeight.bold),
      child: _isFullTextDisplayed
          ? Text(text) // Show full text if the boolean is true
          : AnimatedTextKit(
              animatedTexts: [
                TypewriterAnimatedText(
                  text,
                  speed: Duration(milliseconds: 100),
                ),
              ],
              isRepeatingAnimation: false,
            ),
    );
  }
}
