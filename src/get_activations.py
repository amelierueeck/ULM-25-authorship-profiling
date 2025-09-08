import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import pandas as pd

#get the activations for each layer (length: 13) and save them to a file
def get_activations(texts, layer=0, batch_size=64, max_len=256):
    """
    Args:
      texts (list): list of strings
      out_file (str): file to save activations to
    Returns:
      file: npz file with activations for each layer
    """

    activations = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=max_len).to(device)
        except TypeError:
            continue
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  #tuple of 13 [batch size, sequence length, hidden size 768]

        #take [CLS] token (index 0) from the given layer
        act = hidden_states[layer][:,0,:].cpu().numpy()

        activations.append(act)
    return activations

# load splits
train_df = pd.read_csv("data/data_train.csv")
val_df   = pd.read_csv("data/data_val.csv")
test_df  = pd.read_csv("data/data_test.csv")

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

#get the model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
model.eval()

get_activations(train_df["text"].tolist())
get_activations(val_df["text"].tolist())
get_activations(test_df["text"].tolist())