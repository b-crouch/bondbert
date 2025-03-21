from bondbert.ensemble import MajorityVotingEnsemble, MaxLikelihoodEnsemble, BoostingEnsemble
from bondbert.data import load_json, load_id2label
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="../datasets/processed/matscholar_processed_small.json")

# Load mapping between NER labels and numeric IDs
id2label = load_id2label("../datasets/processed/matscholar_id2label.pkl")

# Specify Hugging Face aliases of models to use as ensemble learners
# Here, we use BatteryBERT (trained on battery physics corpus), SciBERT (trained on general science), and BioBERT (trained on biomedicine)
learners = ["batterydata/batterybert-cased", "allenai/scibert_scivocab_cased", "dmis-lab/biobert-base-cased-v1.2"]



# (1) Majority Voting: Each learner in the ensemble casts a vote for the predicted label; most votes wins
# Initialize MajorityVotingEnsemble
voting_ensemble = MajorityVotingEnsemble(learners, id2label)

# Generate predictions via majority voting across the ensemble
voting_results_dict = voting_ensemble.predict(dataset, batch_size=10)



# (2) Likelihood Maximization: Output the label with highest joint predicted probability density across the ensemble
# Initialize MaxLikelihoodEnsemble
likelihood_ensemble = MaxLikelihoodEnsemble(learners, id2label)

# Generate predictions via likelihood maximization across the ensemble
likelihood_results_dict = likelihood_ensemble.predict(dataset, batch_size=10)



# (3) Boosting: Apply a variant of AdaBoost to weight votes in favor of learners that have previously identified labels correctly
# Initialize BoostedEnsemble
boosted_ensemble = BoostingEnsemble(learners, id2label)

# Train the ensemble to learn per-label boosting scores for each learner
boosted_ensemble.train(dataset, batch_size=10)

# Generate predictions via boosting across the ensemble
boosting_results_dict = boosted_ensemble.predict(dataset, batch_size=10)