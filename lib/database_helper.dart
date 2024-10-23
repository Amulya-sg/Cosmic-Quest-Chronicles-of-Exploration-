import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class Exoplanet {
  final String name;
  final String distance;
  final String hostStar;
  final String size;
  final String potentialForLife;
  final String notableFeature;

  Exoplanet({
    required this.name,
    required this.distance,
    required this.hostStar,
    required this.size,
    required this.potentialForLife,
    required this.notableFeature,
  });
}

class DatabaseHelper {
  static final DatabaseHelper _instance = DatabaseHelper._internal();
  factory DatabaseHelper() => _instance;

  static Database? _database;

  DatabaseHelper._internal();

  Future<Database> get database async {
    _database ??= await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    String path = join(await getDatabasesPath(), 'game_database.db');
    print('Database path: $path');
    return await openDatabase(
      path,
      version: 2, // Updated version to add new tables
      onCreate: (db, version) {
        return _createTables(db);
      },
      onUpgrade: (db, oldVersion, newVersion) {
        if (oldVersion < 2) {
          // Upgrade from version 1 to 2 (Adding planet table)
          db.execute('''
            CREATE TABLE planets (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              distance TEXT,
              hostStar TEXT,
              size TEXT,
              potentialForLife TEXT,
              notableFeature TEXT
            )
          ''');
        }
      },
    );
  }

  Future<void> _createTables(Database db) async {
    // User table
    await db.execute('''
      CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        game_state TEXT
      )
    ''');

    // Points table
    await db.execute('''
      CREATE TABLE points (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        points INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
      )
    ''');

    // Planet table
    await db.execute('''
      CREATE TABLE planets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        distance TEXT,
        hostStar TEXT,
        size TEXT,
        potentialForLife TEXT,
        notableFeature TEXT
      )
    ''');
  }

  Future<int> addUser(String username) async {
    final db = await database;
    int userId =
        await db.insert('users', {'username': username, 'game_state': '{}'});
    await createPointsEntry(userId); // Create points entry for new user
    return userId;
  }

  Future<Map<String, dynamic>?> getUser(int userId) async {
    final db = await database;
    final List<Map<String, dynamic>> result = await db.query(
      'users',
      where: 'id = ?',
      whereArgs: [userId],
    );
    return result.isNotEmpty ? result.first : null;
  }

  Future<void> updateGameState(int userId, String gameState) async {
    final db = await database;
    await db.update(
      'users',
      {'game_state': gameState},
      where: 'id = ?',
      whereArgs: [userId],
    );
  }

  Future<void> addPoints(int userId, int pointsToAdd) async {
    final db = await database;
    await db.rawUpdate('''
      UPDATE points 
      SET points = points + ? 
      WHERE user_id = ?
    ''', [pointsToAdd, userId]);
  }

  Future<int> getUserPoints(int userId) async {
    final db = await database;
    final List<Map<String, dynamic>> result = await db.query(
      'points',
      columns: ['points'], // Only fetch 'points' column
      where: 'user_id = ?',
      whereArgs: [userId],
    );
    return result.isNotEmpty ? result.first['points'] : 0;
  }

  Future<void> createPointsEntry(int userId, {int initialPoints = 0}) async {
    final db = await database;
    try {
      await db.insert('points', {'user_id': userId, 'points': initialPoints});
    } catch (e) {
      print('Error creating points entry: $e');
    }
  }

  Future<List<Map<String, dynamic>>> getAllUsers() async {
    final db = await database;
    try {
      final List<Map<String, dynamic>> result = await db.query('users');
      return result;
    } catch (e) {
      print('Error retrieving all users: $e');
      return [];
    }
  }

  // Function to get user by username
  Future<int?> getUserByUsername(String username) async {
    final db = await database;

    // Query the users table for the username
    List<Map<String, dynamic>> result = await db.query(
      'users',
      where: 'username = ?',
      whereArgs: [username],
    );

    // If a user is found, return the userId, otherwise return null
    if (result.isNotEmpty) {
      return result.first['id'];
    } else {
      return null;
    }
  }

  // Add a method to insert a planet
  Future<int> addPlanet(Exoplanet planet) async {
    final db = await database;
    return await db.insert('planets', {
      'name': planet.name,
      'distance': planet.distance,
      'hostStar': planet.hostStar,
      'size': planet.size,
      'potentialForLife': planet.potentialForLife,
      'notableFeature': planet.notableFeature,
    });
  }

  // Add a method to retrieve all planets
  Future<List<Exoplanet>> getAllPlanets() async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query('planets');

    return List.generate(maps.length, (i) {
      return Exoplanet(
        name: maps[i]['name'],
        distance: maps[i]['distance'],
        hostStar: maps[i]['hostStar'],
        size: maps[i]['size'],
        potentialForLife: maps[i]['potentialForLife'],
        notableFeature: maps[i]['notableFeature'],
      );
    });
  }

  // Insert sample data (optional)
  Future<void> insertSamplePlanets() async {
    await addPlanet(Exoplanet(
      name: "Kepler-452b",
      distance: "1,400 light-years",
      hostStar: "Similar to the Sun (G-type star)",
      size: "1.6 times Earth's diameter",
      potentialForLife:
          "In the habitable zone, might have conditions suitable for water.",
      notableFeature:
          "Sometimes referred to as Earth’s 'cousin' due to its size.",
    ));

    await addPlanet(Exoplanet(
      name: "51 Pegasi b",
      distance: "50 light-years (Pegasus constellation)",
      hostStar: "N/A",
      size: "Hot Jupiter",
      potentialForLife: "Likely hydrogen and helium atmosphere.",
      notableFeature: "First exoplanet discovered orbiting a Sun-like star.",
    ));

    await addPlanet(Exoplanet(
      name: "HD 189733 b",
      distance: "64 light years",
      hostStar: "HD 189733",
      size:
          ", HD 189733b is much bigger and hotter than Earth; it's about the size of Jupiter ",
      potentialForLife:
          "HD 189733b is unlikely to support life because it's too hot and its orbit is too close to its star",
      notableFeature:
          "Winds blow up to 5,400 mph, making it one of the most extreme exoplanets​",
    ));

    await addPlanet(Exoplanet(
      name: "GJ 1132 b",
      distance: "39 light years",
      hostStar: "GJ 1132(Red dwarf)",
      size:
          "The planet's diameter is approximately 20% larger than that of the Earth and its mass is estimated at 1.6 times that of Earth ",
      potentialForLife:
          "GJ 1132 b likely has a “secondary atmosphere,” created by volcanic activity after its first hydrogen-helium atmosphere was stripped away by radiation from its star.",
      notableFeature:
          "The rocky exoplanet GJ 1132 b, similar in size and density to Earth, possesses a hazy atmosphere made up of volcanic gases​",
    ));
  }
}
