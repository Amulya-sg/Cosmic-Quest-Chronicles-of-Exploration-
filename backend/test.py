import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the text-generation model (microsoft/Phi-3.5-mini-instruct)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="auto" if torch.cuda.is_available() else None,  # Only map devices if CUDA is available
    torch_dtype="auto",  # Automatically selects dtype based on model config
    cache_dir="./micro-chat",  # Cache directory to save model
    trust_remote_code=True     # Allows using custom model code if defined remotely
)

# Load the tokenizer for text generation
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", cache_dir="./micro-chat")

# Create a text-generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Load the sentence-transformer model for embedding generation
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder="./embed-model")

# Test the text-generation pipeline
generated_text = pipe("Tell me something interesting about exoplanets.", max_length=50, num_return_sequences=1)
print("Generated text:", generated_text[0]['generated_text'])

# Test the embedding model
sample_sentence = "Exoplanets are planets that orbit stars outside our solar system."
embedding = embedding_model.encode(sample_sentence)
print("Generated embedding:", embedding)
