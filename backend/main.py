
import os
import uuid
import faiss
import pickle
import torch
import joblib
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Exoplanet Quiz and Prediction API")

# Add CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store models and data
planets = {}
embedder = None
index = None
id_to_planet = {}
tokenizer = None
model = None
quiz_store = {}  # Stores quiz_id to correct answers
ml_model = None
scaler = None
label_encoder = None

# Pydantic Models for Quiz
class GenerateQuizRequest(BaseModel):
    planet_name: str

class QuestionOption(BaseModel):
    option: str
    text: str

class Question(BaseModel):
    question: str
    options: List[str]

class GenerateQuizResponse(BaseModel):
    quiz_id: str
    questions: List[Question]

class EvaluateQuizRequest(BaseModel):
    quiz_id: str
    user_answers: List[str]

class EvaluateQuizResponse(BaseModel):
    score: int
    total: int

# Pydantic Models for Planet Type Prediction
class PlanetFeatures(BaseModel):
    mass_multiplier: float
    mass_wrt: str
    radius_multiplier: float
    radius_wrt: str
    distance: float
    stellar_magnitude: float
    orbital_radius: float
    discovery_year: int

class PlanetTypePrediction(BaseModel):
    planet_type: str

# Utility Functions for Quiz
def load_and_partition_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into chunks based on double newlines
    chunks = text.strip().split('\n\n')

    planets_dict = {}
    for chunk in chunks:
        lines = chunk.strip().split('\n')
        if lines:
            planet_name = lines[0].strip()  # Assuming first line is planet name
            planet_info = ' '.join(lines[1:]).strip()  # Rest is info
            planets_dict[planet_name] = planet_info

    return planets_dict

def create_faiss_index_func(planets, embedder, index_path='planets_faiss.index', mapping_path='id_to_planet.pkl'):
    planet_names = list(planets.keys())
    planet_texts = list(planets.values())

    # Generate embeddings
    embeddings = embedder.encode(planet_texts, convert_to_numpy=True, show_progress_bar=True)

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index_local = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index_local.add(embeddings)

    # Save the index
    faiss.write_index(index_local, index_path)

    # Create a mapping from index to planet name
    id_to_planet_local = {i: name for i, name in enumerate(planet_names)}
    with open(mapping_path, 'wb') as f:
        pickle.dump(id_to_planet_local, f)

    print(f"FAISS index and mapping saved to '{index_path}' and '{mapping_path}' respectively.")
    return index_local, id_to_planet_local

def load_faiss_index_func(index_path='planets_faiss.index', mapping_path='id_to_planet.pkl'):
    # Load FAISS index
    index_local = faiss.read_index(index_path)

    # Load mapping
    with open(mapping_path, 'rb') as f:
        id_to_planet_local = pickle.load(f)

    return index_local, id_to_planet_local

def retrieve_planet_data_exact(planet_name, planets):
    return planets.get(planet_name, None)

def retrieve_planet_data_similarity(planet_name, embedder, index_local, id_to_planet_local, planets, top_k=1):
    query_embedding = embedder.encode([planet_name], convert_to_numpy=True)

    distances, indices = index_local.search(query_embedding, top_k)

    closest_id = indices[0][0]
    closest_planet = id_to_planet_local[closest_id]
    closest_distance = distances[0][0]

    planet_info = planets.get(closest_planet, None)

    return closest_planet, planet_info, closest_distance

def get_planet_info(planet_name, planets, embedder, index_local, id_to_planet_local):
    planet_info = retrieve_planet_data_exact(planet_name, planets)
    if planet_info:
        return planet_name, planet_info
    else:
        closest_planet, planet_info, distance = retrieve_planet_data_similarity(
            planet_name, embedder, index_local, id_to_planet_local, planets
        )
        print(f"Did you mean '{closest_planet}'? (Distance: {distance})")
        return closest_planet, planet_info

def load_generation_model_func(model_path):
    # Load the model and tokenizer using the tested pipeline
    tokenizer_local = AutoTokenizer.from_pretrained(model_path, cache_dir="./micro-chat")
    model_local = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype="auto",
        cache_dir="./micro-chat",
        trust_remote_code=True
    )
    return tokenizer_local, model_local

def generate_quiz_func(planet_name, planet_info, tokenizer_local, model_local, num_questions=5):
    prompt = (
        f"Generate {num_questions} multiple-choice questions (MCQs) about {planet_name} based on the following information:\n\n"
        f"{planet_info}\n\n"
        "Each question should have four options labeled A, B, C, and D with only one correct answer. Clearly indicate the correct answer for each question."
    )

    inputs = tokenizer_local.encode(prompt, return_tensors='pt', truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model_local.generate(
            inputs.to(model_local.device),
            max_length=1500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            early_stopping=True
        )

    quiz = tokenizer_local.decode(outputs[0], skip_special_tokens=True)
    return quiz

def parse_quiz(quiz_text):
    """
    Parses the generated quiz text into questions, options, and correct answers.
    Assumes the quiz is formatted with questions numbered and options labeled A-D,
    followed by "Answer: X"
    """
    questions = []
    correct_answers = []

    # Split the quiz into individual questions
    question_blocks = re.split(r'\d+\.\s', quiz_text)[1:]  # First split is empty

    for block in question_blocks:
        # Split into question and answer
        parts = block.split('Answer:')
        if len(parts) != 2:
            continue  # Skip if format is unexpected

        question_part = parts[0].strip()
        answer_part = parts[1].strip()

        # Extract question
        lines = question_part.split('\n')
        question_text = lines[0].strip()

        # Extract options
        options = []
        option_labels = ['A.', 'B.', 'C.', 'D.']
        for line in lines[1:]:
            line = line.strip()
            for label in option_labels:
                if line.startswith(label):
                    option_text = line[len(label):].strip()
                    options.append(option_text)
                    break

        # Extract correct answer
        correct_answer_label = answer_part.upper()
        label_to_index = {'A':0, 'B':1, 'C':2, 'D':3}
        correct_answer_index = label_to_index.get(correct_answer_label, -1)
        if correct_answer_index == -1:
            continue  # Invalid answer label

        correct_answers.append(correct_answer_label)
        questions.append({
            'question': question_text,
            'options': options
        })

    return questions, correct_answers

# FastAPI Event Handlers
@app.on_event("startup")
def startup_event():
    global planets, embedder, index, id_to_planet, tokenizer, model, quiz_store, ml_model, scaler, label_encoder

    # Paths
    txt_path = r'C:\Users\91799\Desktop\spaceapps\quizdata\data.txt'
    index_path = r'C:\Users\91799\Desktop\spaceapps\planets_faiss.index'
    mapping_path = r'C:\Users\91799\Desktop\spaceapps\id_to_planet.pkl'
    generation_model_path = "microsoft/Phi-3.5-mini-instruct"
    ml_model_path = 'models/random_forest_model.joblib'
    scaler_path = 'models/scaler.joblib'
    label_encoder_path = 'models/label_encoder.joblib'

    # Load and partition data
    planets = load_and_partition_txt(txt_path)
    print("Loaded planet data.")

    # Initialize embedder
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder="./embed-model")
    print("Loaded embedder.")

    # Create or load FAISS index
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        index, id_to_planet = create_faiss_index_func(planets, embedder, index_path, mapping_path)
    else:
        index, id_to_planet = load_faiss_index_func(index_path, mapping_path)
    print("Loaded FAISS index.")

    # Load generation model
    tokenizer, model = load_generation_model_func(generation_model_path)
    print("Loaded generation model.")

    # Load ML model, scaler, and label encoder
    try:
        ml_model = joblib.load(ml_model_path)
        print("Loaded ML model.")
    except Exception as e:
        print(f"Failed to load ML model: {e}")

    try:
        scaler = joblib.load(scaler_path)
        print("Loaded scaler.")
    except Exception as e:
        print(f"Failed to load scaler: {e}")

    try:
        label_encoder = joblib.load(label_encoder_path)
        print("Loaded Label Encoder.")
    except Exception as e:
        print(f"Failed to load Label Encoder: {e}")

@app.post("/generate_quiz", response_model=GenerateQuizResponse)
def generate_quiz(request: GenerateQuizRequest):
    global planets, embedder, index, id_to_planet, tokenizer, model, quiz_store

    planet_name_input = request.planet_name
    retrieved_planet, planet_info = get_planet_info(planet_name_input, planets, embedder, index, id_to_planet)

    if not planet_info:
        raise HTTPException(status_code=404, detail=f"Planet '{planet_name_input}' not found.")

    # Generate quiz
    quiz_text = generate_quiz_func(retrieved_planet, planet_info, tokenizer, model)
    print(f"Generated quiz for planet '{retrieved_planet}'.")

    # Parse quiz to extract questions and correct answers
    questions, correct_answers = parse_quiz(quiz_text)

    if not questions:
        raise HTTPException(status_code=500, detail="Failed to parse generated quiz.")

    # Generate a unique quiz ID
    quiz_id = str(uuid.uuid4())
    quiz_store[quiz_id] = correct_answers  # Store correct answers

    # Prepare response
    response_questions = []
    for q in questions:
        response_questions.append(Question(
            question=q['question'],
            options=q['options']
        ))

    return GenerateQuizResponse(
        quiz_id=quiz_id,
        questions=response_questions
    )

@app.post("/evaluate_quiz", response_model=EvaluateQuizResponse)
def evaluate_quiz(request: EvaluateQuizRequest):
    global quiz_store

    quiz_id = request.quiz_id
    user_answers = request.user_answers

    if quiz_id not in quiz_store:
        raise HTTPException(status_code=404, detail="Quiz ID not found.")

    correct_answers = quiz_store[quiz_id]

    if len(user_answers) != len(correct_answers):
        raise HTTPException(status_code=400, detail="Number of answers does not match the number of questions.")

    # Calculate score
    score = 0
    for user_ans, correct_ans in zip(user_answers, correct_answers):
        if user_ans.upper() == correct_ans.upper():
            score += 1

    total = len(correct_answers)

    # Optionally, remove the quiz from store after evaluation
    del quiz_store[quiz_id]

    return EvaluateQuizResponse(
        score=score,
        total=total
    )

# Endpoint for Planet Type Prediction
@app.post("/predict_planet_type", response_model=PlanetTypePrediction)
def predict_planet_type_endpoint(features: PlanetFeatures):
    global ml_model, scaler, label_encoder

    # Check if models are loaded
    if ml_model is None or scaler is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="ML model or scaler not loaded.")

    # Define mass and radius mappings
    mass_mappings = {'Jupiter': 1898, 'Earth': 1}  # mass in Earth masses
    radius_mappings = {'Jupiter': 69911, 'Earth': 6371}  # radius in km

    # Calculate mass and radius
    mass = features.mass_multiplier * mass_mappings.get(features.mass_wrt, np.nan)
    radius = features.radius_multiplier * radius_mappings.get(features.radius_wrt, np.nan)

    if np.isnan(mass) or np.isnan(radius):
        raise HTTPException(status_code=400, detail="Invalid mass_wrt or radius_wrt value.")

    # Prepare the feature array
    input_features = np.array([[features.distance, features.stellar_magnitude, mass, radius, features.orbital_radius]])

    # Scale the features
    try:
        input_scaled = scaler.transform(input_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in scaling features: {e}")

    # Predict the planet type
    try:
        prediction = ml_model.predict(input_scaled)
        planet_type = label_encoder.inverse_transform(prediction)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {e}")

    return PlanetTypePrediction(planet_type=planet_type)