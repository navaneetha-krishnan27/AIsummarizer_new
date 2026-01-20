from sentence_transformers import SentenceTransformer

# Download the model
print("Downloading model... this may take a minute.")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Save it to a folder named 'local_ml_model'
model.save('./local_ml_model')
print("Model saved successfully to ./local_ml_model")