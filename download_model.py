#!/usr/bin/env python3
"""
Script to download and cache the sentence-transformer model locally within the project directory.
Run this once with internet connection to enable offline usage.
"""

from sentence_transformers import SentenceTransformer
import os

def download_model():
    """
    Download and cache the model for offline usage in the project directory.
    """
    model_name = 'all-MiniLM-L6-v2'
    
    # Create local models directory within the project
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    local_model_path = os.path.join(models_dir, model_name)
    
    print(f"Downloading model '{model_name}'...")
    print("This may take a few minutes depending on your internet connection.")
    print(f"Model will be saved to: {local_model_path}")
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Download and save the model to local directory
        model = SentenceTransformer(model_name, cache_folder=models_dir)
        
        # Save the model explicitly to our local path
        model.save(local_model_path)
        
        # Test the model with a simple sentence
        test_sentence = "This is a test sentence."
        embedding = model.encode(test_sentence)
        
        print(f"✅ Model '{model_name}' downloaded and saved successfully!")
        print(f"Model files are stored in: {local_model_path}")
        print(f"Embedding dimension: {len(embedding)}")
        print("\nYou can now run the document analyst offline.")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("Please check your internet connection and try again.")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Sentence Transformer Model Downloader")
    print("=" * 50)
    
    success = download_model()
    
    if success:
        print("\n🎉 Setup complete! You can now run the document analyst without internet.")
    else:
        print("\n💥 Setup failed. Please fix the errors and try again.")
