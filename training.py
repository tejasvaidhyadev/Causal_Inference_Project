import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from dataloader import CustomDataset
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1024, help="batch size for training")
parser.add_argument('--learning_rate', type=float, default=1e-5, help="learning rate for training")
parser.add_argument('--patience', type=int, default=10, help="patience for early stopping")
parser.add_argument('--num_tpus', type=int, default=1, help="number of TPUs to use")
parser.add_argument('--epochs', type=int, default=100, help="number of epochs to train for")
parser.add_argument('--regularization_coefficient', type=float, default=0.1, help="regularization coefficient for MMD loss")
parser.add_argument('--kernel_bandwidth', type=float, default=0.1, help="kernel bandwidth for MMD loss")
parser.add_argument('--num_labels', type=int, default=2, help="number of labels in the dataset")
parser.add_argument('--max_seq_length', type=int, default=128, help="maximum sequence length for BERT")
args = parser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate
patience = args.patience
num_tpus = args.num_tpus
epochs = args.epochs
regularization_coefficient = args.regularization_coefficient
kernel_bandwidth = args.kernel_bandwidth
num_labels = args.num_labels
max_seq_length = args.max_seq_length


# Define the classification model
class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = bert_module
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_labels)

    def forward(self, input_ids, input_mask, segment_ids):
        _, pooled_output = self.bert(input_ids, input_mask, segment_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# main function
if __name__ == "__main__":
    # Load the tokenizer and create the dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = CustomDataset("data/train.csv", tokenizer, max_seq_length=128)

    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.1, random_state=42)
    # Split the dataset and create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=32, sampler=data.SubsetRandomSampler(train_indices), num_workers=4)
    val_loader = data.DataLoader(train_dataset, batch_size=32, sampler=data.SubsetRandomSampler(val_indices), num_workers=4)

    # Load the BERT model from TensorFlow Hub
    bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H768_A-12/1")

    # Initialize the model, optimizer, and loss function
    model = BertClassifier(num_labels=2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        # Train for one epoch
        model.train()
        for batch in train_loader:
            input_ids, input_mask, segment_ids, labels = batch
            logits = model(input_ids, input_mask, segment_ids)
            loss = criterion(logits, labels)
            mmd_loss = MMDLoss(kernel_bandwidth=10.0)(torch.log_softmax(logits, dim=1))
            loss += regularization_coefficient * mmd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation set
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, input_mask, segment_ids, labels = batch
                logits = model(input_ids, input_mask, segment_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Early stopping based on validation loss
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping after", epoch, "epochs")
                break

    # Evaluate the trained model on the test set
    test_dataset = CustomDataset("data/perturbed_test.csv", tokenizer, max_seq_length=128)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, input_mask, segment_ids, labels = batch
            logits = model(input_ids, input_mask, segment_ids)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    print(classification_report(test_labels, test_preds))
