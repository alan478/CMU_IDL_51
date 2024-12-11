import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer
from tqdm import tqdm
from torchinfo import summary

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

# Define a custom Transformer model
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dropout_rate=0.1):
        super(CustomTransformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dropout=dropout_rate, batch_first=True), num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, dropout=dropout_rate, batch_first=True), num_layers
        )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.dropout(self.embedding(src))
        tgt_emb = self.dropout(self.embedding(tgt))
        memory = self.encoder(src_emb, src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask)
        return self.fc_out(output)

# Define loss functions
def reconstruction_loss(pred, target, criterion):
    return criterion(pred.view(-1, pred.size(-1)), target.view(-1))

def adversarial_loss(discriminator, fake_preds, real_preds):
    real_loss = nn.BCEWithLogitsLoss()(real_preds, torch.ones_like(real_preds))
    fake_loss = nn.BCEWithLogitsLoss()(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss

def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def cross_alignment_loss(content_repr_1, content_repr_2, metric=nn.CosineSimilarity(dim=-1)):
    return 1 - metric(content_repr_1, content_repr_2).mean()

# Training function
def train_custom_model(model, train_loader, criterion, optimizer, discriminator, epochs):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_reconstruction_loss = 0
        total_adversarial_loss = 0
        total_kl_loss = 0
        total_cross_alignment_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Batches")

        for batch_idx, batch in progress_bar:
            src, tgt = batch["input_ids"].to(device), batch["labels"].to(device)
            output = model(src, tgt)
            latent_repr = model.encoder(model.embedding(src))

            L_rec = reconstruction_loss(output, tgt, criterion)
            fake_preds = discriminator(latent_repr.detach().mean(dim=1))
            real_preds = discriminator(latent_repr.detach().mean(dim=1))
            L_adv = adversarial_loss(discriminator, fake_preds, real_preds)
            mu, logvar = torch.mean(latent_repr, dim=1), torch.log(torch.var(latent_repr, dim=1) + 1e-8)
            L_KL = kl_divergence_loss(mu, logvar)
            content_repr_1 = model.encoder(model.embedding(src))
            content_repr_2 = model.encoder(model.embedding(tgt))
            L_cross_align = cross_alignment_loss(content_repr_1, content_repr_2)

            L_total = L_rec + 0.5 * L_adv + 0.1 * L_KL + 0.3 * L_cross_align

            optimizer.zero_grad()
            L_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_reconstruction_loss += L_rec.item()
            total_adversarial_loss += L_adv.item()
            total_kl_loss += L_KL.item()
            total_cross_alignment_loss += L_cross_align.item()

            progress_bar.set_postfix({
                "L_rec": f"{total_reconstruction_loss / (batch_idx + 1):.4f}",
                "L_adv": f"{total_adversarial_loss / (batch_idx + 1):.4f}",
                "L_KL": f"{total_kl_loss / (batch_idx + 1):.4f}",
                "L_cross_align": f"{total_cross_alignment_loss / (batch_idx + 1):.4f}",
            })

        print(f"Epoch {epoch + 1} Summary: L_rec: {total_reconstruction_loss / len(train_loader):.4f}, L_adv: {total_adversarial_loss / len(train_loader):.4f}, L_KL: {total_kl_loss / len(train_loader):.4f}, L_cross_align: {total_cross_alignment_loss / len(train_loader):.4f}")

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model = CustomTransformer(vocab_size=32100, d_model=512, num_layers=4, num_heads=8, dropout_rate=0.1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(custom_model.parameters(), lr=1e-4)

# Define dummy discriminator
class DummyDiscriminator(nn.Module):
    def __init__(self, d_model):
        super(DummyDiscriminator, self).__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.linear(x.mean(dim=1))

discriminator = DummyDiscriminator(d_model=512).to(device)

# Start training
train_custom_model(custom_model, train_loader, criterion, optimizer, discriminator, epochs=5)

# Model summary
src = torch.randint(0, 32100, (16, 128)).to(device)
tgt = torch.randint(0, 32100, (16, 128)).to(device)
summary(custom_model, input_data=(src, tgt))

# Validation
val_dataset = torch.load("/content/style_transfer_val_dataset.pth")
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

custom_model.eval()
val_loss = 0
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Validation Batches"):
        src, tgt = batch["input_ids"].to(device), batch["labels"].to(device)
        output = custom_model(src, tgt)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        val_loss += loss.item()

print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
