from datasets import load_dataset

def get_data_splits(to_csv=False, columns=["text", "age", "gender"]):
    """
    Load the blog authorship corpus dataset and split it into training, validation, and test sets.
    Args:
        to_csv (bool): If True, save the splits as CSV files.
        columns (list): List of columns to select from the dataset.
    Returns:
        tuple: DataFrames for training, validation, and test sets.
    """
    dataset = load_dataset("barilan/blog_authorship_corpus", trust_remote_code=True)
    data_train, data_test = dataset["train"], dataset["validation"] # use the validation split as test
    data_train, data_val = data_train.train_test_split(test_size=0.1).values()
    if columns:
        data_train = data_train.select_columns(columns)
        data_val = data_val.select_columns(columns)
        data_test = data_test.select_columns(columns)
    if to_csv:
        data_train.to_pandas().to_csv("data/data_train.csv", index=False)
        data_val.to_pandas().to_csv("data/data_val.csv", index=False)
        data_test.to_pandas().to_csv("data/data_test.csv", index=False)
    return data_train, data_val, data_test

data_train, data_val, data_test = get_data_splits(to_csv=True)

print("Training set size:", len(data_train))
print("Validation set size:", len(data_val))
print("Test set size:", len(data_test))