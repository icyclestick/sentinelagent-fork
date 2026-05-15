import torch
import os
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import Dataset

LABEL_LIST = ["O", "B-ACTION", "I-ACTION", "B-OBJECT", "I-OBJECT", "B-SCOPE", "I-SCOPE", "B-CONSTRAINTS", "I-CONSTRAINTS"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

class DeBERTaDecomposer:
    def __init__(self, model_name='microsoft/deberta-v3-base', num_labels=9, device=None):
        if device is None:
            if torch.cuda.is_available():
                # Fallback to CPU if VRAM is less than 6GB to prevent OOM
                if torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3:
                    self.device = "cpu"
                else:
                    self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)

    def train(self, token_labeled_data, epochs=5, batch_size=2, output_dir="deberta_decomposer_output"):
        """
        Trains the decomposer.
        token_labeled_data: list of dicts {"tokens": [...], "labels": [...], "text": "..."}
        """
        # Convert to HuggingFace dataset
        dataset_dict = {
            "tokens": [d["tokens"] for d in token_labeled_data],
            "ner_tags": [[LABEL_TO_ID[l] for l in d["labels"]] for d in token_labeled_data]
        }
        dataset = Dataset.from_dict(dataset_dict)

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=128)

            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        # For subwords, we can label them as the same or -100
                        # Standard practice: label as -100 for subwords to compute loss only on first token of word
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="no",
            save_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=10,
            report_to="none",
            fp16=(self.device == "cuda"), # Only use fp16 if running on CUDA
            use_cpu=(self.device == "cpu")
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()

    def decompose(self, instruction):
        """
        Infers token classifications and returns dict.
        """
        self.model.eval()
        tokens = self.tokenizer.tokenize(instruction, add_special_tokens=False)
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        predictions = torch.argmax(outputs.logits, dim=2)
        
        # Mapping back
        word_ids = inputs.word_ids()
        
        extracted = {"ACTION": [], "OBJECT": [], "SCOPE": [], "CONSTRAINTS": []}
        
        current_entity = None
        current_tokens = []
        
        preds = predictions[0].cpu().numpy()
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
                
            pred_id = preds[idx]
            label = ID_TO_LABEL[pred_id]
            token = self.tokenizer.decode(inputs.input_ids[0][idx])
            # Handle special spaces in deberta / sentencepiece
            token = token.strip()
            
            if label.startswith("B-"):
                if current_entity:
                    extracted[current_entity].append(" ".join(current_tokens))
                current_entity = label[2:]
                current_tokens = [token]
            elif label.startswith("I-") and current_entity == label[2:]:
                if token:
                    current_tokens.append(token)
            else:
                if current_entity:
                    extracted[current_entity].append(" ".join(current_tokens))
                    current_entity = None
                    current_tokens = []
                    
        if current_entity:
            extracted[current_entity].append(" ".join(current_tokens))
            
        # Clean up
        def clean_spans(spans):
            res = " ".join(spans)
            # Remove double spaces
            res = " ".join(res.split())
            return res if res else "None"
            
        return {
            "action": clean_spans(extracted["ACTION"]),
            "object": clean_spans(extracted["OBJECT"]),
            "scope": clean_spans(extracted["SCOPE"]),
            "constraints": clean_spans(extracted["CONSTRAINTS"])
        }

    def serialize(self, decomposed_dict):
        """
        Converts decomposed dict to natural language string.
        """
        return f"Action: {decomposed_dict['action']}. Object: {decomposed_dict['object']}. Scope: {decomposed_dict['scope']}. Constraints: {decomposed_dict['constraints']}."

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path, device=None):
        instance = cls(model_name=path, device=device)
        return instance
