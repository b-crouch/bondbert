from bondbert.data import load_json, save_id2label

# Load and re-structure JSON of NER-labeled text samples. Generate mapping between NER labels and numeric IDs
data, id2label = load_json("../datasets/raw/matscholar.json")

# Save data and label mapping to file
data.to_json("../datasets/processed/matscholar_processed_full.json", orient="records")
save_id2label(id2label, "../datasets/processed/matscholar_id2label.pkl")