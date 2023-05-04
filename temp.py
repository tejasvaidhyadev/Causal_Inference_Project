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
parser.add_argument('--dataset', default='synthetic_exp3', help="Directory containing the dataset")
parser.add_argument('--batch_size', type=int, default=80, help="batch size for training")
parser.add_argument('--learning_rate', type=float, default=5e-5, help="learning rate for training")
parser.add_argument('--patience', type=int, default=5, help="patience for early stopping")
parser.add_argument('--num_tpus', type=int, default=1, help="number of TPUs to use")
parser.add_argument('--epochs', type=int, default=1, help="number of epochs to train for")
parser.add_argument('--num_labels', type=int, default=2, help="number of labels in the dataset")
parser.add_argument('--max_seq_length', type=int, default=128, help="maximum sequence length for BERT")
parser.add_argument('--is_marginal_reg', type=bool, default=True, help="whether to use marginal regularization")

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


def main(args):
    tagger_model_dir = 'experiments/' + args.dataset
    prepare_environment(tagger_model_dir, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_parameters(device, args)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    train_loader, val_loader = prepare_data_loaders(tokenizer, args)
    
    num_classes = 2
    model = BertClassifier(bert_model, num_classes)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    train_size = len(train_loader.sampler)
    train_steps_per_epoch = train_size // args.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=args.epochs * train_steps_per_epoch)

    criterion = nn.CrossEntropyLoss()
    regularization_coefficients = [ 0 ]

    test_loader, perturbed_test_loader = prepare_test_data_loaders(tokenizer, args)

    for regularization_coefficient in regularization_coefficients:

        # Evaluate the model on the test sets
        in_domain_accuracy, perturbed_accuracy, label_flip_rate, conditional_mmd_test, conditional_mmd_perturbed_test, diff_prob =  train_and_evaluate_model(args, regularization_coefficient, model, criterion, train_loader, val_loader, test_loader, perturbed_test_loader, optimizer, scheduler, device, tagger_model_dir)


        # Log the results
        log_results(regularization_coefficient, in_domain_accuracy, perturbed_accuracy, label_flip_rate, conditional_mmd_test, conditional_mmd_perturbed_test, diff_prob)

def log_results(regularization_coefficient, in_domain_accuracy, perturbed_accuracy, label_flip_rate, conditional_mmd_test, conditional_mmd_perturbed_test, diff_prob):
    logging.info(f"Regularization Coefficient: {regularization_coefficient}")
    logging.info(f"In-Domain Accuracy: {in_domain_accuracy}")
    logging.info(f"Perturbed Accuracy: {perturbed_accuracy}")
    logging.info(f"Label Flip Rate: {label_flip_rate}")
    logging.info(f"Conditional MMD test: {conditional_mmd_test}")
    logging.info(f"Conditional MMD perturbed test: {conditional_mmd_perturbed_test}")
    logging.info(f"Prob Difference: {diff_prob}")
    logging.info("------------------------------------------------")

def prepare_environment(tagger_model_dir, args):
    if not os.path.exists(tagger_model_dir):
        os.makedirs(tagger_model_dir, exist_ok=True)
    set_logger(os.path.join(tagger_model_dir, 'train.log'))
    
    args_json_path = os.path.join(tagger_model_dir, "args.json")
    with open(args_json_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Args saved to {args_json_path}")


def log_parameters(device, args):
    logging.info("device: {}".format(device))
    logging.info("batch_size: {}".format(args.batch_size))
    logging.info("learning_rate: {}".format(args.learning_rate))
    logging.info("patience: {}".format(args.patience))
    logging.info("num_tpus: {}".format(args.num_tpus))
    logging.info("epochs: {}".format(args.epochs))
    logging.info("num_labels: {}".format(args.num_labels))
    logging.info("max_seq_length: {}".format(args.max_seq_length))
    logging.info("is_marginal_reg: {}".format(args.is_marginal_reg))

def prepare_data_loaders(tokenizer, args):
    train_dataset = CustomDataset("data/train.csv", tokenizer, max_seq_length=128)
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.1, random_state=42)
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=data.SubsetRandomSampler(train_indices), num_workers=4)
    val_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=data.SubsetRandomSampler(val_indices), num_workers=4)
    return train_loader, val_loader

def prepare_test_data_loaders(tokenizer, args):
    test_dataset = CustomDataset("data/test.csv", tokenizer, max_seq_length=64)
    perturbed_test_dataset = CustomDataset("data/perturbed_test.csv", tokenizer, max_seq_length=64)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    perturbed_test_loader = data.DataLoader(perturbed_test_dataset, batch_size=args.batch_size, num_workers=4)
    return test_loader, perturbed_test_loader

def evaluate_test_sets(args, model, test_loader, perturbed_test_loader, device, regularization_coefficient):
    # Evaluate the model on the original test set
    model.eval()
    test_preds, test_labels, test_logits, test_zs, test_pooling_outputs = evaluate_one_epoch_test(model, test_loader, device)

    # Calculate in-domain accuracy
    in_domain_accuracy = np.mean(np.array(test_labels) == np.array(test_preds))

    # Evaluate the model on the perturbed test set
    model.eval()
    perturbed_test_preds, perturbed_test_labels, perturbed_test_logits, perturbed_test_zs, perturbed_test_pooling_outputs = evaluate_one_epoch_test(model, perturbed_test_loader, device)

    # Calculate perturbed accuracy
    perturbed_accuracy = np.mean(np.array(perturbed_test_labels) == np.array(perturbed_test_preds))

    # Calculate the rate of predicted label flips
    label_flip_rate = calculate_label_flips(test_preds, perturbed_test_preds)

    # Calculate conditional MMD (implementation depends on the specific MMD loss used)
    diff_prob = np.mean(np.array(test_preds) == 1) - np.mean(np.array(perturbed_test_preds) == 1)
    mmd_loss = MMDLoss(kernel_bandwidth=None)
    mmd_loss.to(device)

    conditional_mmd_test = compute_regularization_term(model, mmd_loss, torch.tensor(test_pooling_outputs).to(device), torch.tensor(test_zs).to(device), torch.tensor(test_labels).to(device), args.is_marginal_reg) 
    conditional_mmd_perturbed_test = compute_regularization_term(model, mmd_loss, torch.tensor(perturbed_test_pooling_outputs).to(device), torch.tensor(perturbed_test_zs).to(device), torch.tensor(perturbed_test_labels).to(device), args.is_marginal_reg) 

    return in_domain_accuracy, perturbed_accuracy, label_flip_rate, conditional_mmd_test, conditional_mmd_perturbed_test, diff_prob


def train_and_evaluate_model(args, regularization_coefficient, model, criterion, train_loader, val_loader, test_loader, perturbed_test_loader, optimizer, scheduler, device, tagger_model_dir):
    
    best_val_loss = float('inf')
    patience_counter = 0
    model_save_path = os.path.join(tagger_model_dir, f"model_reg_coeff_{regularization_coefficient}.pt")

    for epoch in range(args.epochs):
        train_one_epoch(args, epoch, model, criterion, train_loader, optimizer, scheduler, device, regularization_coefficient)

        val_loss, val_preds, val_labels = evaluate_one_epoch(args, epoch, model, criterion, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save the model
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved to {model_save_path}")

        else:
            patience_counter += 1

            if patience_counter >= args.patience:
                logging.info(f"Early stopping after {epoch} epochs")
                break

    # Evaluate the model on the original test set and perturbed test set
    evaluate_test_sets(args, model, test_loader, perturbed_test_loader, device, regularization_coefficient)

def train_one_epoch(args, epoch, model, criterion, train_loader, optimizer, scheduler, device, regularization_coefficient):
    model.train()
    train_loss_avg = RunningAverage()

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Training)")
    for batch in train_loader_tqdm:
        input_ids, input_mask, segment_ids, labels, Z = batch
        input_ids, input_mask, segment_ids, labels, Z = input_ids.to(device), input_mask.to(device), segment_ids.to(device), labels.to(device), Z.to(device)

        pooled_output, logits = model(input_ids, input_mask)

        loss = criterion(logits, labels)
        
        # Compute the marginal regularization term
        mmd_loss = MMDLoss(kernel_bandwidth=None)
        mmd_loss.to(device)
        reg_term = compute_regularization_term(model, mmd_loss, pooled_output, Z, labels, args.is_marginal_reg)
        
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
    logging.info("Epoch {}/{} (Training): loss = {:.5f}".format(epoch+1, args.epochs, train_loss_avg()))

def evaluate_one_epoch_test(model, data_loader, device):
    model.eval()

    preds = []
    labels = []
    logits = []
    zs = []
    pooling_outputs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, input_mask, segment_ids, batch_labels, z = batch
            input_ids, input_mask, segment_ids, batch_labels, z = input_ids.to(device), input_mask.to(device), segment_ids.to(device), batch_labels.to(device), z.to(device)

            pooled_output, batch_logits = model(input_ids, input_mask)
            batch_preds = torch.argmax(batch_logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            logits.extend(batch_logits.cpu().numpy())
            zs.extend(z.cpu().numpy())
            pooling_outputs.extend(pooled_output.cpu().numpy())

    return preds, labels, logits, zs, pooling_outputs

def evaluate_one_epoch(args, epoch, model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Validation)")
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

    val_loss /= len(val_loader)
    
    # log the validation loss
    logging.info("Epoch {}/{} (Validation): loss = {:.5f}".format(epoch+1, args.epochs, val_loss))
    
    # perturbed_test_preds, perturbed_test_labels, perturbed_test_logits, perturbed_test_zs, perturbed_test_pooling_outputs
    return val_loss, val_preds, val_labels


if __name__ == "__main__":

    args = parser.parse_args()
    main(args)
