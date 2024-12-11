import pandas as pd
import json
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch

# Function to load and process data
def load_and_process_data(data_dir):
    attr_file = f"{data_dir}/fader.attr"
    review_file = f"{data_dir}/fader.review"

    # Load attributes and reviews
    attributes = pd.read_csv(attr_file, sep="\t", header=None)
    with open(review_file, "r") as f:
        reviews = f.readlines()

    # Ensure the number of rows matches
    assert len(attributes) == len(reviews), "Mismatch between attributes and reviews!"

    # Define mappings
    gender_map = {0: "Male", 1: "Female", 2: "Unknown"}
    sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    binary_map = {0: "No", 1: "Yes"}

    # Attribute column indices based on `fader.attr`
    attribute_indices = {
        "Gender": 0,
        "Stars": 1,
        "Sentiment": 2,
        "Restaurant": 3,
        "Asian Food": 4,
        "American Food": 5,
        "Mexican Food": 6,
        "Bar Food": 7,
        "Dessert Food": 8
    }

    def map_attributes(row):
        """Convert numeric attributes to descriptive labels."""
        return {
            "Gender": gender_map[row[0]],
            "Stars": row[1],
            "Sentiment": sentiment_map[row[2]],
            "Restaurant": binary_map[row[3]],
            "Asian Food": binary_map[row[4]],
            "American Food": binary_map[row[5]],
            "Mexican Food": binary_map[row[6]],
            "Bar Food": binary_map[row[7]],
            "Dessert Food": binary_map[row[8]],
        }

    # Create formatted pairs
    formatted_data = []
    for idx, row in attributes.iterrows():
        attr_dict = map_attributes(row)
        attr_string = ", ".join([f"{key}: {value}" for key, value in attr_dict.items()])
        review_text = reviews[idx].strip()

        formatted_data.append({
            "input": f"[STYLE: {attr_string}] {review_text}",
            "output": review_text  # Replace with style-transferred text if available
        })

    return formatted_data

# Process both datasets
data_dirs = ["data/amazon", "data/yelp"]
all_data = []
for data_dir in data_dirs:
    all_data.extend(load_and_process_data(data_dir))

# Save to JSON file
with open("formatted_data.json", "w") as f:
    json.dump(all_data, f, indent=4)

# Load the formatted data
with open("formatted_data.json", "r") as f:
    formatted_data = json.load(f)

# Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Tokenize input-output pairs
inputs = [entry["input"] for entry in formatted_data]
outputs = [entry["output"] for entry in formatted_data]

input_encodings = tokenizer(
    inputs, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
)
output_encodings = tokenizer(
    outputs, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
)

# Define PyTorch dataset
class StyleTransferDataset(Dataset):
    def __init__(self, input_encodings, output_encodings):
        self.input_encodings = input_encodings
        self.output_encodings = output_encodings

    def __len__(self):
        return len(self.input_encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_encodings["input_ids"][idx],
            "attention_mask": self.input_encodings["attention_mask"][idx],
            "labels": self.output_encodings["input_ids"][idx]
        }

# Create dataset
dataset = StyleTransferDataset(input_encodings, output_encodings)
torch.save(dataset, "style_transfer_dataset.pth")

# Create validation dataset
from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# Save validation dataset
torch.save(val_dataset, "style_transfer_val_dataset.pth")
