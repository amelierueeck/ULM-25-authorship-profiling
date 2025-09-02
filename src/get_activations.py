import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd

#get the activations for each layer (length: 13) and save them to a file
def get_activations(texts, out_file, batch_size=16, max_len=256):
    """
    Args:
      texts (list): list of strings
      out_file (str): file to save activations to
    Returns:
      file: npz file with activations for each layer
    """

    first_batch = True
    print("Starting get_activations…", flush=True)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=max_len).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  #tuple of 13 [batch size, sequence length, hidden size 768]

        #take [CLS] token (index 0) from each layer
        batch_layers = [layer_hid[:,0,:].cpu().numpy() for layer_hid in hidden_states]

        # doing this to mitigate RAM issues
        if first_batch:
          np.savez_compressed(out_file, **{f"layer{idx}": arr for idx, arr in enumerate(batch_layers)})
          first_batch = False
          print("First batch done", flush=True)

        else:
          existing = dict(np.load(out_file))
          updated = {f"layer{idx}": np.concatenate([existing[f"layer{idx}"], arr], axis=0) for idx, arr in enumerate(batch_layers)}
          np.savez_compressed(out_file, **updated)

        if i % 10000 == 0:
          print(f"Processed {i} texts", flush=True)

    print(f"Saved activations to {out_file}")
    return out_file

# load splits
train_df = pd.read_csv("data_train.csv")
val_df   = pd.read_csv("data_val.csv")
test_df  = pd.read_csv("data_test.csv")

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

get_activations(train_df["text"].tolist(), "train_layers.npz")
get_activations(val_df["text"].tolist(), "val_layers.npz")
get_activations(test_df["text"].tolist(), "test_layers.npz")