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
import logging
from tqdm import tqdm
import json
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

# Make it gpu enable code.

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='synthetic_exp2', help="Directory containing the dataset")
parser.add_argument('--batch_size', type=int, default=128, help="batch size for training")
parser.add_argument('--learning_rate', type=float, default=5e-4, help="learning rate for training")
parser.add_argument('--patience', type=int, default=5, help="patience for early stopping")
parser.add_argument('--num_tpus', type=int, default=1, help="number of TPUs to use")
parser.add_argument('--epochs', type=int, default=5, help="number of epochs to train for")
parser.add_argument('--kernel_bandwidth', type=float, default=0, help="kernel bandwidth for MMD loss")
parser.add_argument('--num_labels', type=int, default=2, help="number of labels in the dataset")
parser.add_argument('--max_seq_length', type=int, default=64, help="maximum sequence length for BERT")
parser.add_argument('--is_marginal_reg', type=bool, default=True, help="whether to use marginal regularization")

args = parser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate
patience = args.patience
num_tpus = args.num_tpus
epochs = args.epochs
kernel_bandwidth = args.kernel_bandwidth
num_labels = args.num_labels
max_seq_length = args.max_seq_length
is_marginal_reg = args.is_marginal_reg

# Define the classification model
    
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BertClassifier, self).__init__()
        self.bert_model = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return pooled_output, logits


def calculate_label_flips(original_preds, perturbed_preds):
    assert len(original_preds) == len(perturbed_preds)
    flips = sum(original_preds[i] != perturbed_preds[i] for i in range(len(original_preds)))
    return flips / len(original_preds)

def compute_marginal_regularization(model, mmd_loss, pooled_output, Z):
    f_X_z0 = pooled_output[Z == 0]
    f_X_z0 = model.classifier(f_X_z0)

    f_X_z1 = pooled_output[Z == 1]
    f_X_z1 = model.classifier(f_X_z1)
    return mmd_loss(f_X_z0, f_X_z1)

def compute_conditional_regularization(model, mmd_loss, pooled_output, Z, Y, y_value):
    f_X_z0_y = pooled_output[(Z == 0) & (Y == y_value)]    
    f_X_z0_y = model.classifier(f_X_z0_y)
    f_X_z1_y = pooled_output[(Z == 1) & (Y == y_value)]
    f_X_z1_y = model.classifier(f_X_z1_y)

    return mmd_loss(f_X_z0_y, f_X_z1_y)

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
        
def compute_regularization_term(model, mmd_loss, pooled_output, Z, labels, is_marginal_reg):
    """
    Compute the regularization term based on the specified type (marginal or conditional).

    Parameters:
    - mmd_loss: The MMD loss function.
    - pooled_output: The pooled output from the model.
    - Z: The latent variable.
    - labels: The labels for the data.
    - is_marginal_reg: A boolean flag indicating whether to compute the marginal regularization term.

    Returns:
    - reg_term: The computed regularization term.
    """

    # Compute the marginal regularization term
    if is_marginal_reg:
        marginal_reg = compute_marginal_regularization(model, mmd_loss, pooled_output, Z)
        reg_term = marginal_reg
    else:
        # Compute the conditional regularization terms
        cond_reg_y0 = compute_conditional_regularization(model, mmd_loss, pooled_output, Z, labels, 0)

        # Compute the conditional regularization terms
        cond_reg_y1 = compute_conditional_regularization(model, mmd_loss, pooled_output, Z, labels, 1)
        
        reg_term = cond_reg_y0 + cond_reg_y1

    return reg_term

# main function
if __name__ == "__main__":

    tagger_model_dir = 'experiments/' + args.dataset
    if not os.path.exists(tagger_model_dir):
        os.makedirs(tagger_model_dir, exist_ok=True)
    set_logger(os.path.join(tagger_model_dir, 'train.log'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args_json_path = os.path.join(tagger_model_dir, "args.json")
    
    with open(args_json_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Args saved to {args_json_path}")

    logging.info("device: {}".format(device))
    logging.info("batch_size: {}".format(batch_size))
    logging.info("learning_rate: {}".format(learning_rate))
    logging.info("patience: {}".format(patience))
    logging.info("num_tpus: {}".format(num_tpus))
    logging.info("epochs: {}".format(epochs))
    logging.info("kernel_bandwidth: {}".format(kernel_bandwidth))
    logging.info("num_labels: {}".format(num_labels))
    logging.info("max_seq_length: {}".format(max_seq_length))
    logging.info("is_marginal_reg: {}".format(is_marginal_reg))

    logging.info("creating the datasets...")
    # Load the tokenizer and create the dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    train_dataset = CustomDataset("data/train.csv", tokenizer, max_seq_length=128)
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.1, random_state=42)
    # Split the dataset and create data loaders
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, sampler=data.SubsetRandomSampler(train_indices), num_workers=4)
    val_loader = data.DataLoader(train_dataset, batch_size=batch_size, sampler=data.SubsetRandomSampler(val_indices), num_workers=4)

    # Initialize the model, optimizer, and loss function
    num_classes = 2  # Number of sentiment classes (e.g., positive and negative)
    logging.info("Creating BERT Model...")
    logging.info("Number of classes: {}".format(num_classes))
    logging.info("bert_model: {}".format(bert_model))

    model = BertClassifier(bert_model, num_classes)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_size = len(train_loader.sampler)
    train_steps_per_epoch = train_size // batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=epochs * train_steps_per_epoch)

    criterion = nn.CrossEntropyLoss()
    regularization_coefficients = [0.0]

    # Create the data loaders for the test and perturbed test sets
    test_dataset = CustomDataset("data/test.csv", tokenizer, max_seq_length=64)
    perturbed_test_dataset = CustomDataset("data/perturbed_test.csv", tokenizer, max_seq_length=64)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    perturbed_test_loader = data.DataLoader(perturbed_test_dataset, batch_size=batch_size, num_workers=4)

    # Run experiments for each regularization coefficient
    for regularization_coefficient in regularization_coefficients:
    # Train the model with the current regularization coefficient
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            # Train for one epoch
            model.train()
            train_loss_avg = RunningAverage()

            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)")
            for batch in train_loader_tqdm:
                input_ids, input_mask, segment_ids, labels, Z = batch
                input_ids, input_mask, segment_ids, labels, Z = input_ids.to(device), input_mask.to(device), segment_ids.to(device), labels.to(device), Z.to(device)

                pooled_output, logits = model(input_ids, input_mask)

                loss = criterion(logits, labels)
                
                # Compute the marginal regularization term
                mmd_loss = MMDLoss(kernel_bandwidth=10.0)
                mmd_loss.to(device)
                reg_term = compute_regularization_term(model, mmd_loss, pooled_output, Z, labels, is_marginal_reg)                
                loss += regularization_coefficient * reg_term

                train_loss_avg.update(loss.item())
                # log the loss
                train_loader_tqdm.set_postfix(loss='{:05.3f}'.format(train_loss_avg()))
                optimizer.zero_grad()
                loss.backward()
                
                # gradient clipping
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)


                optimizer.step()
                scheduler.step()
            # log the loss for each epoch
            logging.info("Epoch {}/{} (Training): loss = {:.5f}".format(epoch+1, epochs, train_loss_avg()))

            # Evaluate on the validation set
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)")
            val_loss_avg = RunningAverage()
            with torch.no_grad():
                for batch in val_loader_tqdm:
                    input_ids, input_mask, segment_ids, labels, Z = batch
                    input_ids, input_mask, segment_ids, labels, Z = input_ids.to(device), input_mask.to(device), segment_ids.to(device), labels.to(device), Z.to(device)

                    pooled_output, logits = model(input_ids, input_mask)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    val_loss_avg.update(loss.item())
                    # log the loss
                    val_loader_tqdm.set_postfix(loss='{:05.3f}'.format(val_loss_avg()))
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # Early stopping based on validation loss
            val_loss /= len(val_loader)
            # log the validation loss
            logging.info("Epoch {}/{} (Validation): loss = {:.5f}".format(epoch+1, epochs, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping after {epoch} epochs")
                    break
        # Save the model
        model_save_path = os.path.join(tagger_model_dir, f"model_reg_coeff_{regularization_coefficient}.pt")
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model saved to {model_save_path}")
        
        # Load the saved model
        
        #model_load_path = os.path.join(tagger_model_dir, f"model_reg_coeff_{regularization_coefficient}.pt")
        #model.load_state_dict(torch.load(model_load_path))
        #logging.info(f"Model loaded from {model_load_path}")

        # Evaluate the model on the original test set
        model.eval()
        test_preds = []
        test_labels = []
        test_logits = []
        test_zs = []
        test_pooling_outputs = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, input_mask, segment_ids, labels, z_test = batch
                input_ids, input_mask, segment_ids, labels, z_test = input_ids.to(device), input_mask.to(device), segment_ids.to(device), labels.to(device), z_test.to(device)

                pooled_output_test, logits = model(input_ids, input_mask)
                preds = torch.argmax(logits,dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_logits.extend(logits.cpu().numpy())
                test_zs.extend(z_test.cpu().numpy())
                test_pooling_outputs.extend(pooled_output_test.cpu().numpy())

        # Calculate in-domain accuracy
        in_domain_accuracy = np.mean(np.array(test_labels) == np.array(test_preds))

        # Evaluate the model on the perturbed test set
        model.eval()
        perturbed_test_preds = []
        perturbed_test_labels = []
        perturbed_test_logits = []
        perturbed_test_zs = []
        perturbed_test_pooling_outputs = []

        with torch.no_grad():
            for batch in perturbed_test_loader:
                input_ids, input_mask, segment_ids, labels, z_perturbed = batch
                input_ids, input_mask, segment_ids, labels, z_perturbed = input_ids.to(device), input_mask.to(device), segment_ids.to(device), labels.to(device), z_perturbed.to(device)

                pooled_output_perturbed, logits = model(input_ids, input_mask)
                preds = torch.argmax(logits, dim=1)
                perturbed_test_preds.extend(preds.cpu().numpy())
                perturbed_test_labels.extend(labels.cpu().numpy())
                perturbed_test_logits.extend(logits.cpu().numpy())
                perturbed_test_zs.extend(z_perturbed.cpu().numpy())
                perturbed_test_pooling_outputs.extend(pooled_output_perturbed.cpu().numpy())

        # Calculate perturbed accuracy
        perturbed_accuracy = np.mean(np.array(perturbed_test_labels) == np.array(perturbed_test_preds))

        # Calculate the rate of predicted label flips
        label_flip_rate = calculate_label_flips(test_preds, perturbed_test_preds)

        # Calculate conditional MMD (implementation depends on the specific MMD loss used)
        diff_prob = np.mean(np.array(test_preds) == 1) - np.mean(np.array(perturbed_test_preds) == 1)
        
        conditional_mmd_test = compute_regularization_term(model, mmd_loss, torch.tensor(test_pooling_outputs).to(device), torch.tensor(test_zs).to(device), torch.tensor(test_labels).to(device), is_marginal_reg) 
        conditional_mmd_perturbed_test = compute_regularization_term(model, mmd_loss, torch.tensor(perturbed_test_pooling_outputs).to(device), torch.tensor(perturbed_test_zs).to(device),torch.tensor(perturbed_test_labels).to(device), is_marginal_reg) 

        # Report the results
        logging.info(f"Regularization Coefficient: {regularization_coefficient}")
        logging.info(f"In-Domain Accuracy: {in_domain_accuracy}")
        logging.info(f"Perturbed Accuracy: {perturbed_accuracy}")
        logging.info(f"Label Flip Rate: {label_flip_rate}")
        logging.info(f"Conditional MMD test: {conditional_mmd_test}")
        logging.info(f"Conditional MMD perturbed test: {conditional_mmd_perturbed_test}")
        logging.info(f"Prob Difference: {diff_prob}")
        logging.info("------------------------------------------------")
