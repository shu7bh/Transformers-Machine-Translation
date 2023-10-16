# Transformer for Translation from English to French from Scratch

This repository has the code for Transformer in `task.py`.

The file `task.py` goes through all the steps:
- Preprocessing the data
    - Loading the data
    - Preprocessing the data
    - Creating the vocabulary
    - Creating the Dataset
- Training the Model
    - Creating the Model
    - Training the Model
    - Evaluating the Model
    - Saving the Model
    - Loading the Model
    - Evaluating the Model on the Test Set
    - Running BLEU Score

There is a parameter `DIR` in both the files which is the path to the directory where you want to save the best models.

The `data` directory should be in the same directory as the code. It should create the following files:
- `train.en`: The English sentences for training
- `train.fr`: The French sentences for training
- `dev.en`: The English sentences for validation
- `dev.fr`: The French sentences for validation
- `test.en`: The English sentences for testing
- `test.fr`: The French sentences for testing

The `train.en` and the `test.csv` have to be present within a directory `data` in the same directory as the code.

### Main Libraries used in the code:

    PyTorch       -> create the models
    Pandas        -> load the data
    NLTK          -> tokenize the data
    WandB         -> log the metrics
    tqdm          -> for progress bars
    torchinfo     -> for model summary
    torchtext     -> for BLEU score
    icecream      -> for debugging

### Steps to run the code:

1. Clone the repository
2. Create a virtual environment
3. Install the requirements
4. Run the code

```bash
python task.py
```