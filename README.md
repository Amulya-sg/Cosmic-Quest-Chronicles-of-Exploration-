# Welcome to our project for NASA SPACE APPS HACKATHON 2024

This is a flutter based application, which is a 2d gameplay and also an educational resource to learn about exoplanets, you can run the app without server file loading to access other parts of app. We have SQlite database as well in the app. This app can run without network and it needs net only for ml model implementation. We have currently implemented ml model in our app and not rag but we do have the rag model ready in working condition
we would recommend you to download SQLITE just in case as the database is stored locally when you run it(not necessarily)

If you need the server code as well:

You need to run the fastend API inorder to access predict_planet model
You can access all other pages without running server, you just need server for MYPLANET i.e predictive model of planet

Steps to clone the app:
1. git clone https://github.com/Harika-Raju/cosmic_quest.git

Steps to run server:
remove the backend folder from flutter project and store it locally on your device. Let the backend folder and flutter project be separate
now in terminal, using cd command, locate to the path of directory where you store your backend folder

1)run python environment
   ```bash
   python -m venv venv_name
   ```
2)Activate it
   ```bash
   Venv_name\Scripts\activate
   ```bash

3) install all requirements
   ```bash
   pip install requirements.txt
   ```

4) Run test.py
   ```bash
   python test.py
   ```
   Running test.py will download and save the models (llm,sentence transformers) in the directory mentioned
   
6) Run ml.py
   ```bash
   python ml.py
   ```
   Then running ml.py will train and save the random forest model

after everything is downloaded without errors, run main.py
NOTE:PLEASE CHANGE PATHS IN main.py and test.py ACCORDING TO YOUR SYSTEM
Note: Before running it , change the paths in main.py to your laptop path where the files are there for data.txt, planets_faiss.index, id_to_planet.pkl and run it

after running it , run the server in this way 
Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   host should be your IP address and run it 

after you run the server, go to myplanetintro.dart and change the backendUrl: 'http://YOUR_IP_ADDRESS:8000 in navigation part of code
this way you will be able to use predict the planet model

Note: this app is accessible only in landscape mode
                





