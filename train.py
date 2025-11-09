import torch
from torch import nn, optim

with open('data', 'r') as file:
    data = file.read()
    data = data.split()

voc = sorted(list(set(data)))
word2index = {word: i for i, word in enumerate(voc)}
index2word = {i: word for word, i in word2index.items()}

x_train = torch.tensor([word2index[word] for word in data[:-1]], dtype=torch.long)
y_train = torch.tensor([word2index[word] for word in data[1:]], dtype=torch.long)

class LexiFlow(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.mlp(out)
        return out 

vocab_size = len(word2index)
embed_size = 64 if vocab_size < 1000 else 128 if vocab_size < 10000 else 256
hidden_size = 128 if vocab_size < 1000 else 256 if vocab_size < 10000 else 512
learning_rate = 0.001
epochs = 150

model = LexiFlow(vocab_size, embed_size, hidden_size)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    total_loss = 0
    optimizer.zero_grad()
    loss = criterion(model(x_train), y_train)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

def predict(text, next_words=1):
    model.eval()
    
    if isinstance(text, str):
        text = text.split()
    input_indices = torch.tensor([word2index[word] for word in text], dtype=torch.long).unsqueeze(0)  # [1, seq_len]

    predicted_words = []

    with torch.no_grad():
        for _ in range(next_words):
            output = model(input_indices)
            last_word_logits = output[0, -1, :]
            prob = torch.softmax(last_word_logits, dim=0)
            next_index = torch.argmax(prob).item()
            next_word = index2word[next_index]
            predicted_words.append(next_word)

            input_indices = torch.cat([input_indices, torch.tensor([[next_index]])], dim=1)

    return " ".join(predicted_words)

print(predict("hello how are"))