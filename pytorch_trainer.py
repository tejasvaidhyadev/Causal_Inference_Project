# This scirpt is port of trainer.py to drop dependency from tf.hub and use huggingface transformers instead.
# Author: Tejas Vaidhya
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from dataloader import CustomDataset
from utils import *
import argparse
import os 

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
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_labels)

    def forward(self, input_ids, input_mask, segment_ids):
        outputs = self.bert(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

def calculate_label_flips(original_preds, perturbed_preds):
    assert len(original_preds) == len(perturbed_preds)
    flips = sum(original_preds[i] != perturbed_preds[i] for i in range(len(original_preds)))
    return flips / len(original_preds)


model_save_dir = "saved_models"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir, exist_ok=True)

# main function
if __name__ == "__main__":
    # Load the tokenizer and create the dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = CustomDataset("data/train.csv", tokenizer, max_seq_length=128)

    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.1, random_state=42)
    # Split the dataset and create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=32, sampler=data.SubsetRandomSampler(train_indices), num_workers=4)
    val_loader = data.DataLoader(train_dataset, batch_size=32, sampler=data.SubsetRandomSampler(val_indices), num_workers=4)

    # Initialize the model, optimizer, and loss function
    model = BertClassifier(num_labels=num_labels)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    regularization_coefficients = [0.0]
    test_dataset = CustomDataset("data/test.csv", tokenizer, max_seq_length=128)
    perturbed_test_dataset = CustomDataset("data/perturbed_test.csv", tokenizer, max_seq_length=128)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    perturbed_test_loader = data.DataLoader(perturbed_test_dataset, batch_size=batch_size, num_workers=4)

    # Run experiments for each regularization coefficient
    for regularization_coefficient in regularization_coefficients:
        # Initialize the model, optimizer, and loss function
        model = BertClassifier(num_labels=2)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        # Train the model with the current regularization coefficient
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

        # Evaluate the model on the original test set
        model.eval()
        test_preds = []
        test_labels = []
        test_logits = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, input_mask, segment_ids, labels = batch
                logits = model(input_ids, input_mask, segment_ids)
                preds = torch.argmax(logits,dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_logits.extend(logits.cpu().numpy())

        # Calculate in-domain accuracy
        in_domain_accuracy = np.mean(np.array(test_labels) == np.array(test_preds))

        # Evaluate the model on the perturbed test set
        model.eval()
        perturbed_test_preds = []
        perturbed_test_labels = []
        perturbed_test_logits = []
        with torch.no_grad():
            for batch in perturbed_test_loader:
                input_ids, input_mask, segment_ids, labels = batch
                logits = model(input_ids, input_mask, segment_ids)
                preds = torch.argmax(logits, dim=1)
                perturbed_test_preds.extend(preds.cpu().numpy())
                perturbed_test_labels.extend(labels.cpu().numpy())
                perturbed_test_logits.extend(logits.cpu().numpy())

        # Calculate perturbed accuracy
        perturbed_accuracy = np.mean(np.array(perturbed_test_labels) == np.array(perturbed_test_preds))

        # Calculate the rate of predicted label flips
        label_flip_rate = calculate_label_flips(test_preds, perturbed_test_preds)

        # Calculate conditional MMD (implementation depends on the specific MMD loss used)
        # conditional_mmd = calculate_conditional_mmd(test_logits, perturbed_test_logits)
        # Note: The implementation of the calculate_conditional_mmd function is not provided here.
        # Please implement this function based on your specific MMD loss.
        

        # Report the results
        print(f"Regularization Coefficient: {regularization_coefficient}")
        print(f"In-Domain Accuracy: {in_domain_accuracy}")
        print(f"Perturbed Accuracy: {perturbed_accuracy}")
        print(f"Label Flip Rate: {label_flip_rate}")
        # print(f"Conditional MMD: {conditional_mmd}")  # Uncomment this line after implementing calculate_conditional_mmd
        print("------------------------------------------------")
