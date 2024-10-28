import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
from torch import nn
import streamlit as st
import json
import re

# Load vocabulary mappings (ensure one-time loading)
with open("word_to_index.json", "r") as f:
    word_to_index = json.load(f)


with open("index_to_word.json", "r") as f:
    index_to_word = json.load(f)
    index_to_word = {int(k): v for k, v in index_to_word.items()}


vocab_size = len(word_to_index)  # Make sure it matches training vocab size

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the improved MLP model
class ImprovedMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, context_size, activation_function):
        super(ImprovedMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        x = self.dropout1(self.activation_function(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Helper functions for text processing
def clean_and_tokenize(text):
    if not text.strip():
        return []
    text = text.replace('start', ' start ').replace('end', ' end ')
    text = re.sub(r'([.,!?])', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip().lower().split()

def words_to_indices(words):
    return [word_to_index.get(word, word_to_index['pad']) for word in words]

# Model loading function
def load_model(context_size, embedding_dim, activation_fn, random_seed):
    model = ImprovedMLP(vocab_size, embedding_dim, hidden_dim=1024, dropout_rate=0.3,
                        context_size=context_size, activation_function=activation_fn).to(device)
    model_path = f'models/model_context_{context_size}_emb_{embedding_dim}_act_{activation_fn.__name__}_seed_{random_seed}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Text generation function
def generate_text(model, start_sequence, num_words, temperature=1.0):
    model.eval()
    generated = list(start_sequence)
    for _ in range(num_words):
        input_seq = torch.tensor(generated[-context_size:], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_seq)
        logits = output.squeeze(0) / temperature
        next_word_idx = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
        generated.append(next_word_idx)
        if index_to_word[next_word_idx] == 'end':
            break
    return ' '.join(index_to_word[idx] for idx in generated if index_to_word[idx] != 'pad')

# Streamlit UI
st.title("Next Word Prediction App")
st.write("Generate text using a next-word prediction MLP model.")

input_text = st.text_input("Enter the starting sequence of words:", value="Mix milk and cream")
context_size = st.selectbox("Context Size", [5, 10], index=1)
embedding_dim = st.selectbox("Embedding Dimension", [32, 64, 128], index=1)
activation_fn_name = st.selectbox("Activation Function", ["tanh", "leaky_relu"], index=0)
random_seed = st.selectbox("Random Seed", [42], index=0)
temperature = st.slider("Temperature", min_value=0.5, max_value= 10.0, value=1.0, step=0.1)
num_words = st.slider("Number of Words to Generate", min_value=10, max_value=100, value=50, step=5)

# Mapping for activation functions
activation_fn_map = {"tanh": torch.tanh, "leaky_relu": F.leaky_relu}
activation_fn = activation_fn_map[activation_fn_name]

# Load and cache the selected model
model = load_model(context_size, embedding_dim, activation_fn, random_seed)

# Tokenize and process input text
start_sequence_words = clean_and_tokenize(input_text)
start_sequence_indices = words_to_indices(start_sequence_words)

# Pad the sequence to match context size
if len(start_sequence_indices) < context_size:
    start_sequence_indices = [word_to_index['pad']] * (context_size - len(start_sequence_indices)) + start_sequence_indices

# Generate text on button click
if st.button("Generate Text"):
    generated_text = generate_text(model, start_sequence_indices, num_words, temperature)
    st.write("Generated Text:")
    st.write(generated_text)


def calculate_perplexity(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = F.nll_loss(output, targets, reduction='sum')
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def load_test_data(context_size):
    test_data_path = f"test_data_context_{context_size}.pt"
    if os.path.exists(test_data_path):
        X_test, Y_test = torch.load(test_data_path)
        print(f"Loaded test data for context size {context_size} from {test_data_path}")
        return X_test, Y_test
    else:
        raise FileNotFoundError(f"Test data for context size {context_size} not found.")


# Load test data and calculate perplexity
X_test, Y_test = load_test_data(context_size)
if X_test is not None and Y_test is not None:
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=64)
    perplexity_score = calculate_perplexity(model, test_loader)
    st.write(f"Perplexity Score: {perplexity_score:.2f}")
