# import PyPDF2  # Not needed since data is in TXT
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import pickle
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
# import torch

# def load_and_partition_txt(txt_path):
#     with open(txt_path, 'r', encoding='utf-8') as file:
#         text = file.read()
    
#     # Split the text into chunks based on double newlines
#     chunks = text.strip().split('\n\n')
    
#     # Optional: Further split each chunk into title and content if structured
#     # For example, if each chunk starts with the planet name
#     planets = {}
#     for chunk in chunks:
#         lines = chunk.strip().split('\n')
#         if lines:
#             planet_name = lines[0].strip()  # Assuming first line is planet name
#             planet_info = ' '.join(lines[1:]).strip()  # Rest is info
#             planets[planet_name] = planet_info
    
#     return planets

# def create_faiss_index(planets, embedder, index_path='planets_faiss.index', mapping_path='id_to_planet.pkl'):
#     planet_names = list(planets.keys())
#     planet_texts = list(planets.values())
    
#     # Generate embeddings
#     embeddings = embedder.encode(planet_texts, convert_to_numpy=True, show_progress_bar=True)
    
#     # Normalize embeddings if using cosine similarity (optional)
#     # faiss.normalize_L2(embeddings)
    
#     # Initialize FAISS index
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)  # Use IndexFlatIP for cosine similarity
#     # If using cosine similarity:
#     # index = faiss.IndexFlatIP(dimension)
#     # faiss.normalize_L2(embeddings)
    
#     # Add embeddings to the index
#     index.add(embeddings)
    
#     # Save the index
#     faiss.write_index(index, index_path)
    
#     # Create a mapping from index to planet name
#     id_to_planet = {i: name for i, name in enumerate(planet_names)}
#     with open(mapping_path, 'wb') as f:
#         pickle.dump(id_to_planet, f)
    
#     print(f"FAISS index and mapping saved to '{index_path}' and '{mapping_path}' respectively.")

# def load_faiss_index(index_path='planets_faiss.index', mapping_path='id_to_planet.pkl'):
#     # Load FAISS index
#     index = faiss.read_index(index_path)
    
#     # Load mapping
#     with open(mapping_path, 'rb') as f:
#         id_to_planet = pickle.load(f)
    
#     return index, id_to_planet

# def retrieve_planet_data_exact(planet_name, planets):
#     return planets.get(planet_name, None)

# def retrieve_planet_data_similarity(planet_name, embedder, index, id_to_planet, planets, top_k=1):
#     query_embedding = embedder.encode([planet_name], convert_to_numpy=True)
#     # faiss.normalize_L2(query_embedding)  # If using cosine similarity
    
#     distances, indices = index.search(query_embedding, top_k)
    
#     # Retrieve the closest planet
#     closest_id = indices[0][0]
#     closest_planet = id_to_planet[closest_id]
#     closest_distance = distances[0][0]
    
#     planet_info = planets.get(closest_planet, None)
    
#     return closest_planet, planet_info, closest_distance

# def get_planet_info(planet_name, planets, embedder, index, id_to_planet):
#     # Attempt exact match
#     planet_info = retrieve_planet_data_exact(planet_name, planets)
#     if planet_info:
#         return planet_name, planet_info
#     else:
#         # Fallback to similarity search
#         closest_planet, planet_info, distance = retrieve_planet_data_similarity(
#             planet_name, embedder, index, id_to_planet, planets
#         )
#         print(f"Did you mean '{closest_planet}'? (Distance: {distance})")
#         return closest_planet, planet_info

# def load_generation_model(model_path):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(model_path)
#     return tokenizer, model

# def generate_quiz(planet_name, planet_info, tokenizer, model, num_questions=5):
#     # Prepare the prompt
#     prompt = (
#         f"Generate {num_questions} multiple-choice questions (MCQs) about {planet_name} based on the following information:\n\n"
#         f"{planet_info}\n\n"
#         "Each question should have four options with only one correct answer."
#     )
    
#     # Tokenize the input
#     inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=1024)
    
#     # Generate output
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs,
#             max_length=1500,
#             num_return_sequences=1,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True,
#             early_stopping=True
#         )
    
#     # Decode the generated text
#     quiz = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     return quiz

# def main():
#     # Paths
#     txt_path = r'C:\Users\Dell\Desktop\spaceapps\quizdata\data.txt'  # Replace with your TXT file path
#     model_path = r'C:\Users\Dell\.cache\huggingface\hub\models--microsoft--Phi-3.5-mini-instruct'
#     index_path = 'planets_faiss.index'
#     mapping_path = 'id_to_planet.pkl'
    
#     # Load and partition data
#     planets = load_and_partition_txt(txt_path)
    
#     # Initialize embedder
#     embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
#     # Create FAISS index (only if not already created)
#     import os
#     if not os.path.exists(index_path) or not os.path.exists(mapping_path):
#         create_faiss_index(planets, embedder, index_path, mapping_path)
    
#     # Load FAISS index and mapping
#     index, id_to_planet = load_faiss_index(index_path, mapping_path)
    
#     # Load generation model
#     tokenizer, model = load_generation_model(model_path)
    
#     # User input
#     planet_name = input("Enter the name of the exoplanet: ").strip()
    
#     # Retrieve planet info
#     retrieved_planet, planet_info = get_planet_info(planet_name, planets, embedder, index, id_to_planet)
    
#     if planet_info:
#         # Generate quiz
#         quiz = generate_quiz(retrieved_planet, planet_info, tokenizer, model)
#         print("\nGenerated Quiz:\n")
#         print(quiz)
#     else:
#         print(f"Planet '{planet_name}' not found in the dataset.")

# if __name__ == "__main__":
#     main()
import os
import torch
import faiss
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

def load_and_partition_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into chunks based on double newlines
    chunks = text.strip().split('\n\n')

    planets = {}
    for chunk in chunks:
        lines = chunk.strip().split('\n')
        if lines:
            planet_name = lines[0].strip()  # Assuming first line is planet name
            planet_info = ' '.join(lines[1:]).strip()  # Rest is info
            planets[planet_name] = planet_info

    return planets

def create_faiss_index(planets, embedder, index_path='planets_faiss.index', mapping_path='id_to_planet.pkl'):
    planet_names = list(planets.keys())
    planet_texts = list(planets.values())

    # Generate embeddings
    embeddings = embedder.encode(planet_texts, convert_to_numpy=True, show_progress_bar=True)

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index.add(embeddings)

    # Save the index
    faiss.write_index(index, index_path)

    # Create a mapping from index to planet name
    id_to_planet = {i: name for i, name in enumerate(planet_names)}
    with open(mapping_path, 'wb') as f:
        pickle.dump(id_to_planet, f)

    print(f"FAISS index and mapping saved to '{index_path}' and '{mapping_path}' respectively.")

def load_faiss_index(index_path='planets_faiss.index', mapping_path='id_to_planet.pkl'):
    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load mapping
    with open(mapping_path, 'rb') as f:
        id_to_planet = pickle.load(f)

    return index, id_to_planet

def retrieve_planet_data_exact(planet_name, planets):
    return planets.get(planet_name, None)

def retrieve_planet_data_similarity(planet_name, embedder, index, id_to_planet, planets, top_k=1):
    query_embedding = embedder.encode([planet_name], convert_to_numpy=True)

    distances, indices = index.search(query_embedding, top_k)

    closest_id = indices[0][0]
    closest_planet = id_to_planet[closest_id]
    closest_distance = distances[0][0]

    planet_info = planets.get(closest_planet, None)

    return closest_planet, planet_info, closest_distance

def get_planet_info(planet_name, planets, embedder, index, id_to_planet):
    planet_info = retrieve_planet_data_exact(planet_name, planets)
    if planet_info:
        return planet_name, planet_info
    else:
        closest_planet, planet_info, distance = retrieve_planet_data_similarity(
            planet_name, embedder, index, id_to_planet, planets
        )
        print(f"Did you mean '{closest_planet}'? (Distance: {distance})")
        return closest_planet, planet_info

def load_generation_model(model_path):
    # Load the model and tokenizer using the tested pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="./micro-chat")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype="auto",
        cache_dir="./micro-chat",
        trust_remote_code=True
    )
    return tokenizer, model

def generate_quiz(planet_name, planet_info, tokenizer, model, num_questions=5):
    prompt = (
        f"Generate {num_questions} multiple-choice questions (MCQs) about {planet_name} based on the following information:\n\n"
        f"{planet_info}\n\n"
        "Each question should have four options with only one correct answer."
    )

    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=1500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            early_stopping=True
        )

    quiz = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return quiz

def main():
    # Paths
    txt_path = r'C:\Users\Dell\Desktop\spaceapps\quizdata\data.txt'
    model_path = "microsoft/Phi-3.5-mini-instruct"
    # index_path = 'planets_faiss.index'
    # mapping_path = 'id_to_planet.pkl'
    index_path = r'C:\Users\Dell\Desktop\spaceapps\planets_faiss.index'
    mapping_path = r'C:\Users\Dell\Desktop\spaceapps\id_to_planet.pkl'


    # Load and partition data
    planets = load_and_partition_txt(txt_path)

    # Initialize embedder
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder="./embed-model")

    # Create FAISS index if not already created
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        create_faiss_index(planets, embedder, index_path, mapping_path)


    # Load FAISS index and mapping
    index, id_to_planet = load_faiss_index(index_path, mapping_path)

    # Load generation model
    tokenizer, model = load_generation_model(model_path)

    # User input
    planet_name = input("Enter the name of the exoplanet: ").strip()

    # Retrieve planet info
    retrieved_planet, planet_info = get_planet_info(planet_name, planets, embedder, index, id_to_planet)

    if planet_info:
        # Generate quiz
        quiz = generate_quiz(retrieved_planet, planet_info, tokenizer, model)
        print("\nGenerated Quiz:\n")
        print(quiz)
    else:
        print(f"Planet '{planet_name}' not found in the dataset.")

if __name__ == "__main__":
    main()
