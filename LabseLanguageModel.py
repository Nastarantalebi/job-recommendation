import torch
from transformers import BertModel, BertTokenizerFast
import pandas as pd

class EmbeddingModelClass:
    def __init__(self) -> None:
        self.model_dir = "./models"  # Define the local language models directory
        self.embedding_model = None
        self.tokenizer = None
        self.df_embeddings = None

    def _load_embedding_model(self):
        try:
            self.embedding_model = BertModel.from_pretrained(
                f"{self.model_dir}/LaBSE_BertModel"
            )
            self.tokenizer = BertTokenizerFast.from_pretrained(
                f"{self.model_dir}/LaBSE_BertTokenizerFast"
            )
        except OSError:
            print("Error: Language models not found. Downloading...")
            # Download the models if not found locally
            self.embedding_model = BertModel.from_pretrained("setu4993/LaBSE")
            self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
            # Save the models for later use
            self.embedding_model.save_pretrained(f"{self.model_dir}/LaBSE_BertModel")
            self.tokenizer.save_pretrained(f"{self.model_dir}/LaBSE_BertTokenizerFast")
        self.embedding_model.eval()  # Correctly reference the model

    def get_embeddings(self, titles_list):
        self._load_embedding_model()
        all_embeddings = []  # Create a list to store embeddings
        
        for title in titles_list:
            inputs = self.tokenizer(title, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            embeddings = outputs.pooler_output.detach().numpy().squeeze()
            all_embeddings.append(embeddings)  # Append to the list

        # Create DataFrame from the list of embeddings
        df_embeddings = pd.DataFrame(all_embeddings)
        
        return df_embeddings
