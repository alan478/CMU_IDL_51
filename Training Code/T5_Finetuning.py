import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved dataset
loaded_dataset = torch.load("/content/style_transfer_dataset.pth")

# Inspect the dataset
print(f"Loaded Dataset Size: {len(loaded_dataset)}")
print(f"Sample Data: {loaded_dataset[0]}")

# Define a custom PyTorch Dataset
class StyleTransferDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["input_ids"].clone().detach().long(),
            "labels": self.data[idx]["labels"].clone().detach().long()
        }

# Create DataLoader
train_loader = DataLoader(
    StyleTransferDataset(loaded_dataset),
    batch_size=16,
    shuffle=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
print(f"Tokenizer Vocabulary Size: {tokenizer.vocab_size}")

# Initialize T5 model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Define optimizer and criterion
optimizer = optim.Adam(t5_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training function
def train_t5_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Batches")
        
        for batch_idx, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{total_loss / (batch_idx + 1):.4f}"})

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

# Training the model
train_t5_model(t5_model, train_loader, optimizer, criterion, epochs=5)

# Save the fine-tuned model
t5_model.save_pretrained("./fine_tuned_t5_model")
tokenizer.save_pretrained("./fine_tuned_t5_model")

# Validation
val_dataset = torch.load("/content/style_transfer_val_dataset.pth")
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

t5_model.eval()
val_loss = 0
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Validation Batches"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = t5_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        val_loss += loss.item()

print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
