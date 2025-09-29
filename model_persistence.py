"""
MODEL PERSISTENCE - How to Save and Load Trained Models

This shows you how to make your models "remember" what they learned!
"""

import torch
import pickle
import os
from basic_neural_language_model import VeryBasicLanguageModel

class PersistentLanguageModel(VeryBasicLanguageModel):
    """
    Enhanced version that can save and load its learned knowledge
    """
    
    def save_model(self, filepath="my_trained_model.pkl"):
        """
        Save the entire trained model to disk
        """
        if self.model is None:
            print("❌ No model to save! Train the model first.")
            return False
            
        # Package everything the model needs to remember
        model_data = {
            'model_state_dict': self.model.state_dict(),  # The learned weights
            'char_to_idx': self.char_to_idx,              # Character mapping
            'idx_to_char': self.idx_to_char,              # Reverse mapping
            'vocab_size': self.vocab_size,                # Vocabulary size
            'model_architecture': {                       # Model structure info
                'vocab_size': self.vocab_size,
                'hidden_size': 128  # Default from create_model
            }
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Model saved successfully to: {filepath}")
            print(f"📁 File size: {os.path.getsize(filepath) / 1024:.1f} KB")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath="my_trained_model.pkl"):
        """
        Load a previously trained model from disk
        """
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore the vocabulary
            self.char_to_idx = model_data['char_to_idx']
            self.idx_to_char = model_data['idx_to_char']
            self.vocab_size = model_data['vocab_size']
            
            # Recreate the model architecture
            self.create_model(hidden_size=model_data['model_architecture']['hidden_size'])
            
            # Load the learned weights
            self.model.load_state_dict(model_data['model_state_dict'])
            
            print(f"✅ Model loaded successfully from: {filepath}")
            print(f"📚 Vocabulary size: {self.vocab_size}")
            print(f"🧠 Parameters: {sum(p.numel() for p in self.model.parameters())}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def train_and_save(self, text, model_name="smart_model", epochs=500):
        """
        Train the model and automatically save it
        """
        print(f"🚀 Training model: {model_name}")
        
        # Train the model
        losses = self.train(text, epochs=epochs)
        
        # Save the trained model
        filepath = f"{model_name}.pkl"
        self.save_model(filepath)
        
        return losses, filepath

def demonstrate_persistence():
    """
    Show how to save and load models
    """
    print("=" * 60)
    print("🧠 MODEL PERSISTENCE DEMONSTRATION")
    print("=" * 60)
    
    # Training data
    training_text = """
    hello world how are you today
    the cat sat on the mat
    i love programming and learning
    python is a great language
    machine learning is fascinating
    neural networks can learn patterns
    this is a simple example
    artificial intelligence is amazing
    """
    
    # Step 1: Train a new model
    print("\n📚 STEP 1: Training a new model...")
    model1 = PersistentLanguageModel()
    losses, model_file = model1.train_and_save(training_text, "shakespeare_bot", epochs=300)
    
    # Test the trained model
    print("\n🎭 Testing freshly trained model:")
    result1 = model1.generate("hello", length=30, temperature=0.8)
    print(f"Generated: {result1}")
    
    # Step 2: Create a new model instance and load the saved model
    print("\n💾 STEP 2: Loading the saved model into a new instance...")
    model2 = PersistentLanguageModel()
    
    print("Before loading - model2 has no knowledge:")
    if model2.model is None:
        print("❌ Model2 is empty (no training)")
    
    # Load the saved model
    success = model2.load_model(model_file)
    
    if success:
        print("\n🎭 Testing loaded model (should be identical):")
        result2 = model2.generate("hello", length=30, temperature=0.8)
        print(f"Generated: {result2}")
        
        print("\n✅ SUCCESS! The model remembered its training!")
        print("💡 You can now close your program and restart it later,")
        print("   and the model will still remember what it learned!")

def save_load_example():
    """
    Simple example of saving and loading
    """
    print("\n" + "=" * 50)
    print("💾 SIMPLE SAVE/LOAD EXAMPLE")
    print("=" * 50)
    
    # Quick training
    model = PersistentLanguageModel()
    text = "the quick brown fox jumps over the lazy dog"
    
    print("🏃 Quick training...")
    model.train(text, epochs=200)
    
    # Save
    model.save_model("quick_model.pkl")
    
    # Load in new instance
    new_model = PersistentLanguageModel()
    new_model.load_model("quick_model.pkl")
    
    # Test both
    print("\nOriginal model:")
    print(model.generate("the", length=20))
    
    print("\nLoaded model:")
    print(new_model.generate("the", length=20))

def model_versioning_example():
    """
    Show how to save different versions of your model
    """
    print("\n" + "=" * 50)
    print("📚 MODEL VERSIONING")
    print("=" * 50)
    
    base_text = "hello world python programming"
    model = PersistentLanguageModel()
    
    # Train and save different versions
    versions = [
        (100, "model_v1_basic.pkl"),
        (300, "model_v2_better.pkl"),
        (500, "model_v3_best.pkl")
    ]
    
    for epochs, filename in versions:
        print(f"\n🚀 Training {filename} for {epochs} epochs...")
        model.train(base_text, epochs=epochs)
        model.save_model(filename)
        
        # Test this version
        result = model.generate("hello", length=15)
        print(f"Result: {result}")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_persistence()
    save_load_example()
    model_versioning_example()
    
    print("\n" + "=" * 60)
    print("🎉 MODEL PERSISTENCE COMPLETE!")
    print("=" * 60)
    print("💡 Key Takeaways:")
    print("✅ Models can save their learned knowledge to disk")
    print("✅ You can load saved models and continue using them")
    print("✅ This allows models to 'remember' across program runs")
    print("✅ You can save different versions as you improve your model")
    print("\n🚀 Next: Try training on your own text and saving it!")
