import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class VeryBasicLanguageModel:
    """
    An extremely simple neural language model for educational purposes.
    This demonstrates the core concepts without complex architecture.
    """
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.model = None
        
    def build_vocabulary(self, text):
        """Create a character-level vocabulary"""
        chars = sorted(list(set(text.lower())))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        print(f"Vocabulary: {chars}")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def text_to_numbers(self, text):
        """Convert text to sequence of numbers"""
        return [self.char_to_idx[ch] for ch in text.lower() if ch in self.char_to_idx]
    
    def numbers_to_text(self, numbers):
        """Convert sequence of numbers back to text"""
        return ''.join([self.idx_to_char[num] for num in numbers])
    
    def create_training_data(self, text, sequence_length=10):
        """Create input-output pairs for training"""
        numbers = self.text_to_numbers(text)
        
        inputs = []
        targets = []
        
        # Create sequences: predict next character given previous characters
        for i in range(len(numbers) - sequence_length):
            input_seq = numbers[i:i + sequence_length]
            target_char = numbers[i + sequence_length]
            inputs.append(input_seq)
            targets.append(target_char)
        
        return torch.tensor(inputs), torch.tensor(targets)
    
    def create_model(self, hidden_size=128):
        """Create a simple neural network"""
        class SimpleLM(nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
                self.linear = nn.Linear(hidden_size, vocab_size)
                
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                output = self.linear(lstm_out[:, -1, :])  # Take last output
                return output
        
        self.model = SimpleLM(self.vocab_size, hidden_size)
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(self, text, epochs=1000, learning_rate=0.01):
        """Train the model"""
        print("Building vocabulary...")
        self.build_vocabulary(text)
        
        print("Creating model...")
        self.create_model()
        
        print("Preparing training data...")
        inputs, targets = self.create_training_data(text)
        print(f"Training on {len(inputs)} examples")
        
        # Set up training
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print("Starting training...")
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print("Training completed!")
        return losses
    
    def generate(self, start_text="", length=100, temperature=1.0):
        """Generate new text"""
        if self.model is None:
            print("Model not trained yet!")
            return ""
        
        self.model.eval()
        
        # Start with given text or random character
        if start_text:
            generated = start_text.lower()
        else:
            generated = list(self.char_to_idx.keys())[0]
        
        with torch.no_grad():
            for _ in range(length):
                # Get last 10 characters (or less)
                recent_chars = generated[-10:]
                input_seq = self.text_to_numbers(recent_chars)
                
                # Pad if needed
                while len(input_seq) < 10:
                    input_seq = [0] + input_seq
                
                # Convert to tensor
                input_tensor = torch.tensor([input_seq])
                
                # Get prediction
                logits = self.model(input_tensor)
                
                # Apply temperature and sample
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_char_idx = torch.multinomial(probs, 1).item()
                
                # Add to generated text
                next_char = self.idx_to_char[next_char_idx]
                generated += next_char
        
        return generated

# Simple demonstration
def demo():
    print("=== Very Basic Language Model Demo ===\n")
    
    # Minimal training data
    training_text = """
    hello world how are you today
    the cat sat on the mat
    i love programming and learning
    python is a great language
    machine learning is fascinating
    neural networks can learn patterns
    this is a simple example
    """
    
    # Create and train model
    model = VeryBasicLanguageModel()
    losses = model.train(training_text, epochs=500)
    
    print("\n=== Generated Text Examples ===")
    
    # Generate text with different prompts
    prompts = ["hello", "the", "i", ""]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = model.generate(prompt, length=50, temperature=0.8)
        print(f"Generated: {generated}")

if __name__ == "__main__":
    demo()
