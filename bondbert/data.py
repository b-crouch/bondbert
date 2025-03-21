"""
Utility methods for handling data pre-processing, loading, and tokenization 
"""
import pickle
import pandas as pd
from datasets import DatasetDict
from typing import Union, Optional
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertTokenizerFast
from transformers import DataCollatorForTokenClassification

class TokenizeHandler:
    """
    Class for tokenizing and aligning input datasets: handles subtoken alignment for NER tags and dataloader creation
    Params:
        tokenizer (transformers.BertTokenizer): Tokenizer instance for text tokenization
    """
    def __init__(self, 
                 tokenizer: Union[BertTokenizer, BertTokenizerFast]):
        self.tokenizer = tokenizer
        
    def _tokenize_and_align(self, 
                            dataset: DatasetDict,
                            max_length: Optional[int]=256):
        """
        Tokenize text in `dataset` and align NER tags with subtokens, applying max length truncation, padding, and "ignore" tag (-100)
        assignment for special characters and padding
        Params:
            dataset (DatasetDict): Dataset to be tokenized
            max_length (int, optional): Maximum length of tokenized output
        Return:
            tokenized_inputs (BatchEncoding): 
        """
        # Tokenize input words in `dataset`
        tokenized_inputs = self.tokenizer(dataset["tokens"], is_split_into_words=True, padding="max_length", max_length=max_length)

        # Align NER tags with tokenized structure
        all_tags = dataset["ner_tags"]
        aligned_tags = []
        for i, tags in enumerate(all_tags):
            word_ids = tokenized_inputs.word_ids(i)
            aligned_tags.append(self._align_tags(tags, word_ids))

        tokenized_inputs["ner_tags"] = aligned_tags
        return tokenized_inputs
    
    def _align_tags(self, 
                    tags: list, 
                    word_ids: list):
        """
        Adjust NER tag assignment to match tokenization scheme: this involves duplicating NER labels as needed to match the partitioning of tokens into subtokens
        Special tokens (CLS, SEP, PAD) are assigned -100 and ignored in loss calculations
        Params:
            tags (list): Original list of NER entity labels
            word_ids (list): List of IDs designating the original word from which each token in the tokenized input originated
        Return:
            aligned_tags (list): List of NER entity labels after alignment to match tokenization
        """
        aligned_tags = []
        current_word = None

        # Iterate over word IDs post-tokenization
        for word_id in word_ids:
            # Expand NER tag list as necessary to align with tokenization scheme
            if word_id is None:
                aligned_tags.append(-100)
            elif word_id != current_word:
                aligned_tags.append(tags[word_id])
            else:
                tag = tags[word_id]
                if tag%2 == 1:
                    tag += 1
                aligned_tags.append(tag)
        return aligned_tags
    
    def _tokenize_dataset(self, 
                          dataset: DatasetDict):
        """
        Apply tokenization and alignment to full dataset
        Params:
            dataset (DatasetDict): Dataset to be tokenized
        Return:
            tokenized_dataset (DatasetDict): Tokenized and aligned dataset
        """
        tokenized_dataset = dataset.map(self._tokenize_and_align, batched=True, remove_columns=dataset["train"].column_names)
        return tokenized_dataset
    
    def make_dataloader(self, 
                        dataset: DatasetDict, 
                        batch_size: int, 
                        shuffle: bool=True):
        """
        Tokenizes and aligns dataset; then, generates iterable dataloader object for use in modeling
        Params:
            dataset (DatasetDict): Dataset to be loaded
            batch_size (int): Batch size to use when loading data
            shuffle (bool): If True, shuffles data when loading
        Return:
            dataloader (iter(DatasetDict)): Batched data iterator for tokenized `dataset`
        """
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, return_tensors="pt")
        tokenized_dataset = self._tokenize_dataset(dataset).rename_column("ner_tags", "labels")
        dataloader = iter(DataLoader(tokenized_dataset["train"], shuffle=shuffle, collate_fn=data_collator, batch_size=batch_size))
        return dataloader
    
    def get_vocab(self):
        """
        Retrieve tokenizer vocabulary mapping
        Return:
            vocab (ItemsView): Mapping of (token, ID) pairs from `self.tokenizer` vocabulary
        """
        vocab = self.tokenizer.get_vocab().items()
        return vocab
    
def load_json(filepath: str):
    """
    Load JSON of NER-tagged text data into a DataFrame and construct mapping between entity labels and numeric IDs
    Params:
        filepath (str): String filepath to JSON
    Return:
        entity_data (pd.DataFrame): Formatted DataFrame of NER data
        id2label (dict): Mapping between entity labels and numeric IDs
    """
    raw_data = pd.read_json(filepath, orient='split')
    
    # Extract tokens
    entity_data = raw_data["labels"].apply(lambda token_dict: [token for token in token_dict]).to_frame().rename(columns={"labels":"tokens"})
    
    # Extract entity labels
    entity_data["ner_names"] = raw_data["labels"].apply(lambda token_dict: [token_dict[token] for token in token_dict])
    
    # Create mapping between entity labels and numeric IDs
    id2label = {idx:name for idx, name in enumerate(entity_data["ner_names"].explode().unique())}
    label2id = {id2label[idx]:idx for idx in id2label}

    entity_data["ner_tags"] = entity_data["ner_names"].apply(lambda name_list: [label2id[name] for name in name_list])
     
    return entity_data, id2label

def load_id2label(filepath: str):
    """
    Load pickled mapping between NER labels and numeric IDs into a dictionary
    Params:
        filepath (str): Filepath to pickled mapping
    Return:
        id2label (dict): Mapping between numeric IDs and entity label names
    """
    with open(filepath, 'rb') as handle:
        id2label = pickle.load(handle)
    return id2label

def save_id2label(id2label: dict,
                  filepath: str):
    """
    Save dictionary mapping between NER labels and numeric IDs to a .pkl
    Params:
        id2label (dict): Mapping between numeric IDs and entity label names
        filepath (str): Filepath to pickled mapping
    """
    with open(filepath, 'wb') as handle:
        pickle.dump(id2label, handle, protocol=pickle.HIGHEST_PROTOCOL)