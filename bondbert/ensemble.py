"""
Core functionality to handle multi-learner ensemble
"""
import math
import torch
import evaluate
import numpy as np
from tqdm import tqdm
from collections import deque
from datasets import DatasetDict
from .data import TokenizeHandler
import spacy_alignments as tokenizations
from .learner import Learner, LearnerOutput
from typing import List, Dict, Optional, Union
from transformers import BertTokenizerFast, BertForTokenClassification


class VotingEnsemble:
    """
    Base class for voting-based learner ensembles
    Handles:
        - Model initialization, data loading, and tokenization for each leaner of the ensemble
        - Alignment of tokenization schemes across different learners
        - Conversion between numeric label IDs and their corresponding entity names
    """
    def __init__(self, 
                 learner_names: Union[List[str], Dict[str, str]], 
                 id2label: Dict,
                 device = "cpu"):
        """
        Initialize VotingEnsemble
        Params:
            learner_names (List[Str] or Dict[str:str]): Learners to be used in the ensemble. Specify as either 1) a list of Hugging Face aliases for each learner (to use a base model with no finetuning)
                                                        or 2) a dictionary mapping between Hugging Face aliases and the path to a checkpoint file for a finetuned model
            id2label (Dict): Dictionary mapping between numeric IDs and entity tag labels in dataset
            device (str): Device for hosting models
        """
        self.n_learners = len(learner_names)
        self.learners = []
        self.id2label = id2label
        self.label2id = {id2label[id]:id for id in id2label}
        self.device = torch.device(device)
        
        # Initialize learner models and store in `self.learners`
        for learner_name in learner_names:
            tokenizer = BertTokenizerFast.from_pretrained(learner_name, do_lower_case=False)
            tokenize_handler = TokenizeHandler(tokenizer)
            model = BertForTokenClassification.from_pretrained(learner_name, id2label=self.id2label, label2id=self.label2id, output_hidden_states=True)  
            
            # Load finetuning checkpoints, if provided
            if type(learner_names) == dict:
                finetuned_model_checkpoint = learner_names[learner_name]
                model.load_state_dict(torch.load(finetuned_model_checkpoint, map_location=self.device)["model_state_dict"])   
            
            learner = Learner(learner_name, model, tokenize_handler, self.device)
            self.learners.append(learner)

    def _init_dataloaders(self, 
                          dataset: DatasetDict, 
                          batch_size: int):
        """
        Initialize dataloaders for each learner in the ensemble
        Params:
            dataset (DatasetDict): Dataset to be loaded for model inference
            batch_size (int): Batch size for data loading
        Return:
            n_batches (int): Number of batches stored by dataloaders
        """
        for learner in self.learners:
            learner.init_dataloader(dataset, batch_size)
        n_batches = len(learner.dataloader)
        return n_batches
    
    def _align_outputs_with_ref(self,
                                outputs: LearnerOutput, 
                                ref_outputs: LearnerOutput):
        """
        Align the logits, tokens, and labels under the tokenization scheme of one learner with that of a reference learner
        Applies Myers' algorithm to identify a mapping between the tokenization of `ref_outputs` and that of `outputs`, then, augments the embedding scheme
        of `outputs` to have dimension equal to that of `ref_outputs`. See README for algorithm details
        Params:
            outputs (bondbert.LearnerOutput): LearnerOutput object containing logits, tokens, and labels of learner to be aligned
            ref_outputs (bondbert.LearnerOutput): LearnerOutput object containing logits, tokens, and labels of learner to align against
        Return:
            outputs (bondbert.LearnerOutput): LearnerOutput object post-alignment
        """
        n_tokens = ref_outputs.logits.shape[0]
        # Iterate over sentence examples in the dataset
        for example_idx in range(n_tokens):
            # Run Myers' algorithm to generate scheme to align `outputs` with the tokenization of `ref_outputs`
            alignment_map, _ = tokenizations.get_alignments(outputs.tokens[example_idx], ref_outputs.tokens[example_idx])
            
            # Then, expand the current tokenization of `outputs` using the alignment scheme
            insert_idx = 0
            for token_idx, aligned_token_idx in enumerate(alignment_map):
                if insert_idx >= outputs.tokens.shape[1]:
                    break
                # Consider cases:
                # (1) Current token is padding
                if ref_outputs.tokens[example_idx][token_idx] == "[PAD]":
                    outputs.tokens[example_idx, insert_idx] = "[PAD]"
                    outputs.logits[example_idx, insert_idx] = -100*torch.ones(outputs.logits.shape[2])
                    
                    if outputs.labels is not None:
                        # If ground truth labels are known, also perform alignment on them
                        outputs.labels[example_idx, insert_idx] = -100
                    insert_idx += 1
                
                # (2) Current token maps to one or more tokens in the reference; re-partition to match the cardinality of the reference
                else:
                    cardinality = len(aligned_token_idx)
                    outputs.tokens[example_idx, insert_idx:insert_idx+cardinality] = ref_outputs.tokens[example_idx, aligned_token_idx]
                    outputs.logits[example_idx, insert_idx:insert_idx+cardinality] = ref_outputs.logits[example_idx, token_idx, :]
                    
                    if outputs.labels is not None:
                        # If ground truth labels are known, also perform alignment on them
                        outputs.labels[example_idx, insert_idx:insert_idx+cardinality] = ref_outputs.labels[example_idx, token_idx].item() if \
                                                                                    torch.is_tensor(outputs.labels[example_idx, token_idx]) else outputs.labels[example_idx, token_idx]
                    insert_idx += cardinality
            
            # Pad the remaining entries in the token tensors
            while insert_idx < outputs.tokens.shape[1]:
                outputs.tokens[example_idx, insert_idx] = "[PAD]"
                outputs.labels[example_idx, insert_idx] = -100
                outputs.logits[example_idx, insert_idx, :] = -100*torch.ones(outputs.logits.shape[2])
                insert_idx += 1
            
            return outputs
        
    def _align_output_pair(self, 
                           outputs_1: LearnerOutput,
                           outputs_2: LearnerOutput):
        """
        Mutually align the logits, tokens, and labels under the tokenization scheme of `outputs_1` with those of `outputs_2`. This requires two passes of Myers'
        algorithm to ensure the alignments of either tokenization are mutually consistent
        Params:
            outputs_1 (bondbert.LearnerOutput): LearnerOutput object containing logits, tokens, and labels of first learner to be aligned
            outputs_2 (bondbert.LearnerOutput): LearnerOutput object containing logits, tokens, and labels of second learner to be aligned
        Return:
            outputs_1 (bondbert.LearnerOutput): Aligned outputs of first learner
            outptus_2 (bondbert.LearnerOutput): Aligned outputs of second learner
        """
        # Align outputs of first learner with those of the second
        aligned_outputs_1 = self._align_outputs_with_ref(outputs_1, outputs_2)
        # Then, align outputs of second learner with those of the first
        aligned_outputs_2 = self._align_outputs_with_ref(outputs_2, aligned_outputs_1)
        return aligned_outputs_1, aligned_outputs_2

            
    def _align_all_learners(self):
        """
        Run alignment algorithm to standardize tokenization scheme across *all* learners of the ensemble; this ensures that the logit predictions made by
        each learner are mutually compatible and all reference the same token set
        Return:
            aligned_outputs (collections.deque): Queue of LearnerOutput objects all aligned to share a common tokenization
        """
        outputs_to_align = deque([])
        
        # Iterate over learners and generate predictions for the current batch
        for learner in self.learners:
            assert learner.dataloader is not None, f"{learner} has not yet initialized a DataLoader!" 
            batch = next(learner.dataloader).to(self.device)
            learner_output = learner.get_output(batch)
            outputs_to_align.append(learner_output)
        
        # Calculate number of alignment passes needed to ensure all embeddings are consistent
        n_alignment_passes = 1 + math.floor(self.n_learners/2)
        n_passes_completed = 0
        
        aligned_outputs = deque([])
        # Arbitrarily select the output of one learner as a reference to begin the alignment algorithm
        reference_output = outputs_to_align.popleft()
        aligned_outputs.append(reference_output)
        
        # Run alignment algorithm: iteratively align pairs of learners until embeddings are consistent across the ensemble
        # Two queues are used to track which two learners should be mutually aligned at each iteration
        while len(outputs_to_align) > 0:
            outputs_1 = outputs_to_align.popleft()
            outputs_2 = aligned_outputs[-1]
            outputs_1, outputs_2 = self._align_output_pair(outputs_1, outputs_2)
            if n_passes_completed < n_alignment_passes:
                outputs_to_align.append(aligned_outputs.pop())
            aligned_outputs.append(outputs_1)
            n_passes_completed += 1
        
        return aligned_outputs
        
    def _convert_id2label(self, 
                          predicted_ids: torch.Tensor,
                          true_ids: Optional[torch.Tensor] = None):
        """
        Map numeric entity IDs outputted by learners to their corresponding labels. If ground truth entity IDs are
        also passed, remove "ignore index" (-100) labels while performing this conversion to prepare results
        for use in boosting analysis
        Args:
            predicted_ids (torch.Tensor): Tensor of numeric entity IDs outputted by learner model
            true_ids (torch.Tensor, optinal): Tensor of ground truth numeric entity IDs
        Return:
            predicted_labels (list): List of string labels for the predicted entities
        """
        if true_ids is None:
            predicted_labels = [[self.id2label[p.item()] for p in pred] for pred in predicted_ids]
            return predicted_labels
        else:
            # Training a BoostingEnsemble requires computing the per-entity precision of each learner; preprocess predictions for this calculation by removing "ignore" tags
            true_labels = [[self.id2label[l.item()] for l in label if l != -100] for label in true_ids]
            predicted_labels = [[self.id2label[p.item()] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predicted_ids, true_ids)]
            return predicted_labels, true_labels
    

class MajorityVotingEnsemble(VotingEnsemble):
    """
    Class for named entity recognition using a learning ensemble with winner-takes-all majority voting rule
    """
    def predict(self, 
                dataset: DatasetDict,
                batch_size: int):
        """
        Make entity label predictions in `dataset` using a majority voting ensemble scheme
        Args:
            dataset (DatasetDict): datasets.DatasetDict containing tokens to be analyzed
            batch_size (int): Batch size for loading data
        Return:
            ensemble_predictions (list): List of entity label predictions generated by the voting scheme
            tokens (list): List of tokens corresponding to predicted entity labels; these may differ from the original tokenization in `dataset` due to tokenization
                           alignment across learners of the ensemble
        """
        # Initialize dataloaders for each learner
        n_batches = self._init_dataloaders(dataset, batch_size)
        
        # If `dataset` contains ground truth entity labels, apply alignment to them as well to allow for performance metric calculations
        dataset_contains_ground_truth = "ner_tags" in dataset["train"].features
        true_labels = []
        
        # Iterate over batches to generate entity label predictions
        ensemble_predictions = []
        tokens = []
        with tqdm(total=n_batches) as progress_bar:
            for batch_idx in range(n_batches):
                # Generate the predictions of each learner and align predictions across learners
                aligned_learner_outputs = self._align_all_learners()

                # Collect the per-learner label predictions across the ensemble
                batch_predicted_labels = []
                batch_true_labels = []
                while len(aligned_learner_outputs) > 0:
                    learner_output = aligned_learner_outputs.pop()
                    learner_predicted_labels = torch.argmax(torch.softmax(learner_output.logits, dim=-1), dim=-1)
                    batch_predicted_labels.append(learner_predicted_labels)
                
                # Apply majority voting rule to output the ensemble prediction for the current batch
                ensemble_batch_votes = torch.mode(torch.stack(batch_predicted_labels, dim=-1), dim=-1).values
                
                # Record results
                if dataset_contains_ground_truth:
                    ensemble_batch_predictions, true_batch_labels = self._convert_id2label(ensemble_batch_votes, learner_output.labels)
                    ensemble_predictions.append(ensemble_batch_predictions)
                    true_labels.append(true_batch_labels)
                else:
                    ensemble_batch_predictions = self._convert_id2label(ensemble_batch_votes)
                    ensemble_predictions.append(ensemble_batch_predictions)
                
                # Collect the tokens post-alignment for mapping back to original words
                tokens.append(learner_output.tokens)
                
                progress_bar.update(1)
                
        if dataset_contains_ground_truth:
            result = {"predictions":ensemble_predictions, "true":true_labels, "tokens":tokens}
        else:
            result = {"predictions":ensemble_predictions, "tokens":tokens}
        
        return result
        
    def __repr__(self):
        return f"MajorityVotingEnsemble({self.name})"
        

class MaxLikelihoodEnsemble(VotingEnsemble):
    """
    Class for named entity recognition using a learning ensemble that selects a tag prediction by choosing
    the label with maximum joint predicted density across all learners in the ensemble
    """
    def predict(self, 
                dataset: DatasetDict,
                batch_size: int):
        """
        Make entity label predictions in `dataset` by computing the joint predicted probability density for each label across all learners
        Args:
            dataset (DatasetDict): datasets.DatasetDict containing tokens to be analyzed
            batch_size (int): Batch size for loading data
        Return:
            ensemble_predictions (list): List of entity label predictions generated by the likelihood maximization scheme
            tokens (list): List of tokens corresponding to predicted entity labels; these may differ from the original tokenization in `dataset` due to tokenization
                           alignment across learners of the ensemble
        """
        # Initialize dataloaders for each learner
        n_batches = self._init_dataloaders(dataset, batch_size)
        
        # If `dataset` contains ground truth entity labels, apply alignment to them as well to allow for performance metric calculations
        dataset_contains_ground_truth = "ner_tags" in dataset["train"].features
        true_labels = []
        
        # Iterate over batches to generate entity label predictions
        ensemble_predictions = []
        tokens = []
        with tqdm(total=n_batches) as progress_bar:
            for batch_idx in range(n_batches):
                # Generate the predictions of each learner and align predictions across learners
                aligned_learner_outputs = self._align_all_learners()

                # Compute joint probability of each entity label across all learners
                joint_probability = 1
                while len(aligned_learner_outputs) > 0:
                    learner_output = aligned_learner_outputs.pop()
                    joint_probability *= torch.softmax(learner_output.logits, dim=-1)
                ensemble_batch_votes = torch.argmax(joint_probability, dim=-1)
                
                # Record results
                if dataset_contains_ground_truth:
                    ensemble_batch_predictions, true_batch_labels = self._convert_id2label(ensemble_batch_votes, learner_output.labels)
                    ensemble_predictions.append(ensemble_batch_predictions)
                    true_labels.append(true_batch_labels)
                else:
                    ensemble_batch_predictions = self._convert_id2label(ensemble_batch_votes)
                    ensemble_predictions.append(ensemble_batch_predictions)
                
                # Collect the tokens post-alignment for mapping back to original words
                tokens.append(learner_output.tokens)
                
                progress_bar.update(1)

        if dataset_contains_ground_truth:
            result = {"predictions":ensemble_predictions, "true":true_labels, "tokens":tokens}
        else:
            result = {"predictions":ensemble_predictions, "tokens":tokens}
        
        return result
    
    def __repr__(self):
        return f"MaxLikelihoodEnsemble({self.learners})"
    
class BoostingEnsemble(VotingEnsemble):
    """
    Class for named entity recognition using a learning ensemble that selects tag predictions via an Adaboost variant
    """
    def __init__(self, 
                 learner_names: List[str], 
                 id2label: Dict,
                 device = "cpu"):
        super().__init__(learner_names, id2label, device)
        self.is_trained = False
        
    def _init_boosting_error_metric(self):
        """
        Initialize an evaluation metric to compute per-class errors to generate boosting weights alpha (see README for mathematical formulation)
        """
        for learner in self.learners:
            learner.boosting_error_metric = evaluate.load("seqeval", zero_division=0)
        
    def _calc_alpha(self):
        """
        Compute boosting weights alpha for each learner (see README for mathematical formulation)
        """
        n_labels = len(self.id2label)
        
        for learner in self.learners:
            # Compute per-label precision 
            learner_results = learner.boosting_error_metric.compute()
            alpha_by_label = torch.zeros(len(self.id2label))
            epsilon = 1e-6 # Avoid division by zero errors
            
            for label_idx, label_id in enumerate(self.id2label):
                # Remove match start/end tags B- and I-  to retain only the entity label
                label = self.id2label[label_idx][2:]
                # If label is not present in the current batch, assume precision is 1/n_labels
                precision = learner_results[label]["precision"] if label in learner_results else 1/n_labels
                error = 1 - precision
                alpha_by_label[label_idx] = np.log((1-error+epsilon)/(error+epsilon)) + np.log(n_labels-1)
            
            # Store boosting scores for inference time
            learner.alpha = alpha_by_label
            
    def _calc_boosting_scores(self, predictions_by_learner, alphas_by_learner, dim):
        """
        Weight predictions outputted by each learner by their per-label boosting weight to generate boosting scores
        Args:
            predictions_by_learner (torch.Tensor): Tensor containing numeric label IDs predicted by each learner in the ensemble
            alphas_by_learner (torch.Tensor): Tensor containing boosting weights (alphas) predicted by each learner in the ensemble
            dim (torch.tensor): Tensor containing the dimensions of the tokenized text after alignment across all learners in the ensemble
        Return:
            boosting_scores (torch.Tensor): Per-entity label boosting score for each learner in the ense,ble
        """
        boosting_scores = torch.zeros(dim)
        for label_idx, label_id in enumerate(self.id2label):
            learner_alphas = alphas_by_learner[label_idx, :]
            predicted_label_mask = (predictions_by_learner == label_id)
            learner_alphas_reshaped = learner_alphas.unsqueeze(0).unsqueeze(1).repeat(predicted_label_mask.shape[0], predicted_label_mask.shape[1], 1)
            boosting_scores[:, :, label_idx] = torch.sum(learner_alphas_reshaped * predicted_label_mask, dim=-1)
        return boosting_scores
        
    def train(self,
              dataset: DatasetDict,
              batch_size: int):
        """
        Train ensemble on `dataset` to generate boosting weights alpha
        Args:
            dataset (DatasetDict): datasets.DatasetDict containing tokens to be analyzed
            batch_size (int): Batch size for loading data
        """
        assert "ner_tags" in dataset["train"].features, "Training requires ground truth NER labels! Ensure dataset contains `ner_tags` column"
        
        # Initialize dataloaders and error recording for each learner
        n_batches = self._init_dataloaders(dataset, batch_size)
        self._init_boosting_error_metric()
        
        # Iterate over batches to generate entity label predictions
        with tqdm(total=n_batches) as progress_bar:
            for batch_idx in range(n_batches):
                # Generate the predictions of each learner and align predictions across learners
                aligned_learners = self._align_all_learners()
                
                # Determine entity label predictions for each learner
                while len(aligned_learners) > 0:
                    learner_output = aligned_learners.pop()
                    learner_predicted_ids = torch.argmax(torch.softmax(learner_output.logits, dim=-1), dim=-1)
                    true_ids = learner_output.labels
                    predicted_labels, true_labels = self._convert_id2label(learner_predicted_ids, true_ids)
                    
                    # Update learner error metric with the current batch
                    for learner in self.learners:
                        if learner.name == learner_output.name:
                            learner.boosting_error_metric.add_batch(predictions=predicted_labels, references=true_labels)
                    progress_bar.update(1)

            # Compute boosting weights for each learner
            self._calc_alpha()
        
        self.is_trained = True
            
    def predict(self, 
                dataset: DatasetDict,
                batch_size: int):
        """
        Make entity label predictions on `dataset` by taking the average prediction across the ensemble weighted by Adaboost-type boosted weightings
        Args:
            dataset (DatasetDict): datasets.DatasetDict containing tokens to be analyzed
            batch_size (int): Batch size for loading data
        Return:
            ensemble_predictions (list): List of entity label predictions generated by the boosting scheme
            tokens (list): List of tokens corresponding to predicted entity labels; these may differ from the original tokenization in `dataset` due to tokenization
                           alignment across learners of the ensemble
        """
        assert self.is_trained, "BoostingEnsemble must be trained before prediction! Call .train() first"

        # Initialize dataloaders for all learners in the ensemble
        n_batches = self._init_dataloaders(dataset, batch_size)
        
        # If `dataset` contains ground truth entity labels, apply alignment to them as well to allow for performance metric calculations
        dataset_contains_ground_truth = "ner_tags" in dataset["train"].features
        true_labels = []
        
        # Iterate over batches to generate entity label predictions
        tokens = []
        ensemble_predictions = []
        with tqdm(total=n_batches) as progress_bar:
            for batch_idx in range(n_batches):
                # Generate the predictions of each learner and align predictions across learners
                aligned_learner_outputs = self._align_all_learners()

                # Extract per-entity label predictions and boosting weights for each learner
                batch_alphas = []
                batch_predictions = []
                while len(aligned_learner_outputs) > 0:
                    learner_output = aligned_learner_outputs.pop()
                    
                    # Record the dimension of outputs after alignment algorithm has been run
                    dim_post_alignment = learner_output.logits.shape
                    
                    # Record predictions
                    for learner in self.learners:
                        if learner.name == learner_output.name:
                            learner_alpha = learner.alpha
                            
                    learner_predicted_labels = torch.argmax(torch.softmax(learner_output.logits, dim=-1), dim=-1)
                    batch_alphas.append(learner_alpha)
                    batch_predictions.append(learner_predicted_labels)
                
                # Apply boosting to compute weighted prediction for each learner
                predictions_by_learner = torch.stack(batch_predictions, dim=-1)
                alpha_by_learner = torch.stack(batch_alphas, dim=-1)
                boosting_scores = self._calc_boosting_scores(predictions_by_learner, alpha_by_learner, dim_post_alignment)
                ensemble_batch_votes = torch.argmax(boosting_scores, dim=-1)
                
                # Record results
                if dataset_contains_ground_truth:
                    ensemble_batch_predictions, true_batch_labels = self._convert_id2label(ensemble_batch_votes, learner_output.labels)
                    ensemble_predictions.append(ensemble_batch_predictions)
                    true_labels.append(true_batch_labels)
                else:
                    ensemble_batch_predictions = self._convert_id2label(ensemble_batch_votes)
                    ensemble_predictions.append(ensemble_batch_predictions)
                
                # Collect the tokens post-alignment for mapping back to original words
                tokens.append(learner_output.tokens)
                
                progress_bar.update(1)

        if dataset_contains_ground_truth:
            result = {"predictions":ensemble_predictions, "true":true_labels, "tokens":tokens}
        else:
            result = {"predictions":ensemble_predictions, "tokens":tokens}
        
        return result
    
    def __repr__(self):
        return f"VotingEnsemble({self.learners})"