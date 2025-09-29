"""
SIMPLE SAVE & LOAD EXAMPLE

This shows the basic steps to make your model remember what it learned.
"""

import torch
import pickle
from basic_neural_language_model import VeryBasicLanguageModel

def train_and_save_model():
    """
    Step 1: Train a model and save it
    """
    print("🚀 STEP 1: Training and saving a model...")
    
    # Create and train model
    model = VeryBasicLanguageModel()
    training_text = """
    hello world how are you today
    the cat sat on the mat
    python is amazing for machine learning
    neural networks can learn patterns
    """
    
    # Train the model
    print("Training...")
    model.train(training_text, epochs=300)
    
    # Test before saving
    print("\n🧪 Testing before saving:")
    result = model.generate("hello", length=30)
    print(f"Generated: {result}")
    
    # Save the model
    model_data = {
        'model_state': model.model.state_dict(),
        'char_to_idx': model.char_to_idx,
        'idx_to_char': model.idx_to_char,
        'vocab_size': model.vocab_size
    }
    
    with open('my_smart_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Model saved as 'my_smart_model.pkl'")
    return True

def load_and_test_model():
    """
    Step 2: Load the saved model and test it
    """
    print("\n💾 STEP 2: Loading the saved model...")
    
    # Create a new model instance
    model = VeryBasicLanguageModel()
    
    # Load the saved data
    with open('my_smart_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Restore the vocabulary
    model.char_to_idx = model_data['char_to_idx']
    model.idx_to_char = model_data['idx_to_char']
    model.vocab_size = model_data['vocab_size']
    
    # Recreate the model architecture
    model.create_model()
    
    # Load the learned weights
    model.model.load_state_dict(model_data['model_state'])
    
    print("✅ Model loaded successfully!")
    
    # Test the loaded model
    print("\n🧪 Testing loaded model:")
    result = model.generate("hello", length=30)
    print(f"Generated: {result}")
    
    return model

def main():
    print("=" * 50)
    print("💾 MODEL SAVE & LOAD DEMONSTRATION")
    print("=" * 50)
    
    # Train and save
    train_and_save_model()
    
    # Load and test
    loaded_model = load_and_test_model()
    
    print("\n🎉 SUCCESS!")
    print("The model remembered its training even after being saved and loaded!")
    
    # Show that you can continue using it
    print("\n🔄 You can keep using the loaded model:")
    prompts = ["the", "python", "neural"]
    for prompt in prompts:
        result = loaded_model.generate(prompt, length=20)
        print(f"Prompt '{prompt}': {result}")

if __name__ == "__main__":
    main()
