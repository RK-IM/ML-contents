from config import CFG
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path

import torch
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers import models

def load_sbert(model_path: str,
               sbert_path: str,
               pretrained_model: str = CFG.PRETRAINED_MODEL):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = Path(model_path)
    s_bert_path = model_path / f"{sbert_path}"

    # If trained model saved in local directory, load model. 
    # Otherwise load pretrained model
    if s_bert_path.is_dir():
        s_bert = SentenceTransformer(s_bert_path, device=device)
        print("[INFO] Load custom trained sbert model.")

    else:
        # load Pre-trained Transformer model
        word_embedding_model = models.Transformer(pretrained_model)
        
        # Add Pooling layer
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        # Add Dense layer 
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=512, 
            activation_function=torch.nn.Tanh()
        )
        
        s_bert = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, dense_model],
            device=device
        )

    return s_bert

class SimpleMLC(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=180),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=180, out_features=self.n_classes),
        )

    def forward(self, x):
        return self.classifier(x)