Named Entity Recognition (NER) using BiLSTM in PyTorch
This project implements a Named Entity Recognition (NER) model using a Bi-directional LSTM (BiLSTM) architecture in PyTorch. The model identifies entities such as persons, organizations, locations, etc., in sequences of text.

Project Overview
Named Entity Recognition is a fundamental task in Natural Language Processing (NLP) where the goal is to locate and classify named entities in text into predefined categories.

This implementation uses:

Word embeddings (nn.Embedding)

Bi-directional LSTM (nn.LSTM)

Fully connected layer for tag prediction

Cross-Entropy Loss with masking for padding tokens

Model Architecture
Input Tokens → Embedding Layer → BiLSTM → Dropout → Fully Connected Layer → Tag Predictions
Embedding Layer: Converts token indices into dense vectors.

BiLSTM Layer: Captures forward and backward context.

Dropout Layer: Prevents overfitting.

Fully Connected Layer: Maps LSTM outputs to tag scores.

Dataset Format
The dataset is assumed to be token-level labeled text sequences (e.g., CoNLL 2003-like format). The data is prepared as:

train_data = [
    (["EU", "rejects", "German", "call"], ["B-ORG", "O", "B-MISC", "O"]),
    ...
]
Label padding is done with <PAD>, and padding index is ignored in the loss.
Features
Custom dataset handling

Dynamic label-to-index and word-to-index mapping

Padding and batching

Training and evaluation loops with accuracy and loss tracking

Evaluation
Evaluation is performed on a validation set and returns accuracy and loss per epoch.

Sample output:

Epoch 03/10
	Train Loss: 0.5544 | Train Acc: 82.26%
	 Val. Loss: 0.7119 |  Val. Acc: 78.12%
Metrics Tracking
Accuracy: percentage of correctly predicted tags (excluding padding)

Loss: average Cross-Entropy Loss per token

Test Set Performance (using best saved model):
	Test Loss: 0.7359
	Test Acc: 77.31%

embedding_dim = 100
hidden_dim = 256
n_layers = 2
dropout = 0.3
learning_rate = 0.001
n_epochs = 10


Dependencies
Python 3.7+

PyTorch

tqdm

numpy (optional for preprocessing)
