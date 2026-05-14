import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

class ContrastiveP2Model:
    def __init__(self, model_name='BAAI/bge-large-en-v1.5', device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name

    def train(self, triples, epochs=5, batch_size=2):
        """
        Trains using MultipleNegativesRankingLoss.
        triples: list of (anchor, positive, hard_negative)
        """
        train_examples = []
        for anchor, positive, negative in triples:
            train_examples.append(InputExample(texts=[anchor, positive, negative]))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * epochs * 0.1),
            show_progress_bar=True,
            use_amp=True
        )

    def predict(self, anchor, instruction):
        """
        Returns cosine similarity (float) between anchor and instruction.
        """
        embeddings1 = self.model.encode([anchor], convert_to_tensor=True)
        embeddings2 = self.model.encode([instruction], convert_to_tensor=True)
        
        cosine_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        return cosine_scores[0].item()

    def classify(self, anchor, instruction, thresholds=(0.9, 0.6)):
        """
        Classifies based on thresholds.
        >= 0.9 -> ALIGNED
        0.6 - 0.9 -> AMBIGUOUS
        < 0.6 -> FLAGGED
        """
        score = self.predict(anchor, instruction)
        upper, lower = thresholds
        if score >= upper:
            return "ALIGNED"
        elif score >= lower:
            return "AMBIGUOUS"
        else:
            return "FLAGGED"

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path, device=None):
        instance = cls(model_name=path, device=device)
        return instance
