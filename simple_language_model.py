import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt

class SimpleTokenizer:
    """A basic tokenizer that converts text to numbers and back"""
    
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
    
    def build_vocab(self, texts):
        """Build vocabulary from a list of texts"""
        # Combine all texts and split into words
        all_text = " ".join(texts).lower()
        # Simple tokenization: split by spaces and remove punctuation
        words = re.findall(r'\b\w+\b', all_text)
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Create vocabulary (most common words first)
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        
        for word, count in word_counts.most_common():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Create inverse mapping
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample words: {list(self.vocab.keys())[:10]}")
    
    def encode(self, text):
        """Convert text to list of token IDs"""
        words = re.findall(r'\b\w+\b', text.lower())
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
    
    def decode(self, token_ids):
        """Convert list of token IDs back to text"""
        words = [self.inverse_vocab.get(token_id, "<UNK>") for token_id in token_ids]
        return " ".join(words)

class SimpleTransformer(nn.Module):
    """A very basic transformer-like model"""
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Position embeddings (for a maximum sequence length)
        self.max_seq_len = 100
        self.position_embedding = nn.Embedding(self.max_seq_len, embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = self.dropout(token_emb + pos_emb)
        
        # Create attention mask (to ignore padding tokens)
        attention_mask = (input_ids != 0).float()
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
        
        # Pass through transformer
        transformer_output = self.transformer(embeddings)
        
        # Get predictions for next token
        logits = self.output_layer(transformer_output)
        
        return logits

class SimpleLanguageModel:
    """Main class that combines tokenizer and model"""
    
    def __init__(self, embedding_dim=64, hidden_dim=128, num_heads=4, num_layers=2):
        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.training_losses = []
    
    def prepare_data(self, texts, max_seq_len=50):
        """Prepare training data from texts"""
        # Build vocabulary
        self.tokenizer.build_vocab(texts)
        
        # Initialize model now that we know vocab size
        self.model = SimpleTransformer(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )
        
        # Create training sequences
        input_sequences = []
        target_sequences = []
        
        for text in texts:
            # Encode text
            tokens = [self.tokenizer.vocab["<START>"]] + self.tokenizer.encode(text) + [self.tokenizer.vocab["<END>"]]
            
            # Create input/target pairs (predict next token)
            for i in range(len(tokens) - 1):
                # Take sequences of max_seq_len
                start_idx = max(0, i - max_seq_len + 1)
                input_seq = tokens[start_idx:i+1]
                target_token = tokens[i+1]
                
                # Pad sequence if needed
                if len(input_seq) < max_seq_len:
                    input_seq = [self.tokenizer.vocab["<PAD>"]] * (max_seq_len - len(input_seq)) + input_seq
                
                input_sequences.append(input_seq)
                target_sequences.append(target_token)
        
        return torch.tensor(input_sequences), torch.tensor(target_sequences)
    
    def train(self, texts, epochs=100, learning_rate=0.001, batch_size=32):
        """Train the model on provided texts"""
        print("Preparing data...")
        input_sequences, target_sequences = self.prepare_data(texts)
        
        print(f"Training on {len(input_sequences)} sequences")
        
        # Set up training
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab["<PAD>"])
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(input_sequences), batch_size):
                batch_inputs = input_sequences[i:i+batch_size]
                batch_targets = target_sequences[i:i+batch_size]
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(batch_inputs)
                
                # Get predictions for the last position (next token prediction)
                last_logits = logits[:, -1, :]
                loss = criterion(last_logits, batch_targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            self.training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    
    def generate_text(self, prompt="", max_length=50, temperature=1.0):
        """Generate text given a prompt"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        # Start with prompt or just start token
        if prompt:
            tokens = [self.tokenizer.vocab["<START>"]] + self.tokenizer.encode(prompt)
        else:
            tokens = [self.tokenizer.vocab["<START>"]]
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input (last 50 tokens or less)
                input_tokens = tokens[-50:]
                if len(input_tokens) < 50:
                    input_tokens = [self.tokenizer.vocab["<PAD>"]] * (50 - len(input_tokens)) + input_tokens
                
                input_tensor = torch.tensor([input_tokens])
                
                # Get predictions
                logits = self.model(input_tensor)
                last_logits = logits[0, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop if we hit end token
                if next_token == self.tokenizer.vocab["<END>"]:
                    break
                
                tokens.append(next_token)
        
        # Decode and return
        generated_tokens = tokens[1:]  # Remove start token
        return self.tokenizer.decode(generated_tokens)
    
    def plot_training_loss(self):
        """Plot the training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

def main():
    """Demonstration of the simple language model"""
    print("=== Simple Language Model Demo ===\n")
    
    # Very minimal training data - just a few sentences
    training_texts = [
        "the cat sat on the mat",
        "the dog ran in the park", 
        "the bird flew over the tree",
        "the fish swam in the water",
        "the cat played with the ball",
        "the dog barked at the moon",
        "the bird sang a beautiful song",
        "the fish jumped out of water",
        "cats and dogs are pets",
        "birds can fly high in the sky",
        "water is essential for fish",
        "the sun is bright today",
        "the moon shines at night",
        "trees grow tall and strong",
        "parks are fun places to play"
    ]
    
    # Create and train model
    model = SimpleLanguageModel(embedding_dim=32, hidden_dim=64, num_heads=2, num_layers=2)
    
    print("Training the model...")
    model.train(training_texts, epochs=200, learning_rate=0.001, batch_size=8)
    
    print("\n=== Generation Examples ===")
    
    # Generate some text
    prompts = ["the cat", "the dog", "the bird", ""]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        for i in range(3):
            generated = model.generate_text(prompt, max_length=10, temperature=0.8)
            print(f"Generated {i+1}: {generated}")
    
    # Plot training loss (comment out if matplotlib not available)
    try:
        model.plot_training_loss()
    except:
        print("Matplotlib not available for plotting")

if __name__ == "__main__":
    main()
