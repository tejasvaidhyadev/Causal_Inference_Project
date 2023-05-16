

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import matplotlib.pyplot as plt
#from transformers import BertTokenizer, BertModel

import argparse
import os 
import logging
from tqdm import tqdm
import json
import torch.utils.data as data

pre = ''
'''GOOGLE DRIVE CODE

from google.colab import drive
drive.mount('/content/gdrive')
pre = '/content/gdrive/MyDrive/MILA/Dhanya Stuff/course project/github/'

! pip install transformers

import sys
sys.path.append('/content/gdrive/MyDrive/MILA/Dhanya Stuff/course project/github/')'''


import dataloader
from dataloader import CustomDataset
from utils import MMDLoss, RunningAverage
import fairness_metrics as fm
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

#COMPAS Dataset Loader
class COMPASDataset(data.Dataset):
    def __init__(self, simple_dataset, train, sensitive_feature):
        post = None
        if train:
          post = "_train.csv"
        else:
          post = "_test.csv"
        if simple_dataset:
            data = pre+"data/propublica_data" + post
        else:
            data = pre+"data/raw_filtered_data" + post
        self.data = pd.read_csv(data, skiprows=1)
        self.simple_dataset = simple_dataset
        self.sensitive_feature = sensitive_feature
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.simple_dataset:
            #ignore first column (indexing), and last three columns (decile score, recid)
            x = self.data.iloc[idx, 1:].astype(float)
            if self.sensitive_feature == 'race':
                #get race as protected attribute
                z = self.data.iloc[idx, 4].astype(float)
            elif self.sensitive_feature == 'gender':
                #get gender as protected attribute
                z = self.data.iloc[idx, 9].astype(float)
            else:
                print("SENSATIVE FEATURE NOT AVAILABE YET")
            #only use recid as label, -2 is violent recid, -3 is decile score
            y = self.data.iloc[idx, 0].astype(float)
            return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float), torch.tensor(z, dtype=torch.float)
        else:
            #ignore first column (indexing), and last three columns (decile score, recid)
            x = self.data.iloc[idx, 1:-3].astype(float)
            if self.sensitive_feature == 'race':
                #get race as protected attribute
                z = self.data.iloc[idx, 5].astype(float)
            elif self.sensitive_feature == 'gender':
                #get gender as protected attribute
                z = self.data.iloc[idx, 1].astype(float)
            else:
                print("SENSATIVE FEATURE NOT AVAILABE YET")
            #only use recid as label, -2 is violent recid, -3 is decile score
            y = self.data.iloc[idx, -3].astype(float)
            return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float), torch.tensor(z, dtype=torch.float)

# Define the classification model

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forward(self, x):
        probs = self.classifier(x)
        pred = (probs >= 0.5).squeeze().float()
        return pred, probs

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid()
            )
        self.sigmoid = nn.Sigmoid()
    '''  nn.Flatten(),
      nn.Linear(input_dim, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      
    )'''
    def forward(self, x):
        probs = self.classifier(x)#self.sigmoid(self.classifier(x))
        pred = torch.round(probs).float()
        return pred, probs

class MLPComplex(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLPComplex, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size*2)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        logits = self.output_layer(x)
        probs = self.softmax(logits)
        preds = torch.argmax(probs, dim=1)
        return logits, preds

    def classifier(self, logits):
        probs = self.softmax(logits)
        #preds = torch.argmax(probs)
        return probs


def compute_marginal_regularization(model, mmd_loss, pooled_output, Z):
    #print(Z.shape)
    f_X_z0 = pooled_output[Z == 0]
    #print("pooled_output", pooled_output.shape)
    #print("f_X_z0", f_X_z0.shape)
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
    #print("reg term: ", reg_term)
    return reg_term

# main function

    
def train(regularization_coefficients, is_marginal_reg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_type = "MLPComplex" #"Log Reg" #MLP
    simple_dataset = dataset == 'compas_simple'
    optimizer_str = "Adam"
    sensitive_feature = 'race'

    in_dim = None
    out_dim = 1

    if simple_dataset:
        in_dim = 10
    else:
        in_dim = 15


    train_dataset = COMPASDataset(simple_dataset, True, sensitive_feature)
    
    test_accs = torch.zeros(len(regularization_coefficients))
    test_acc_losses = torch.zeros(len(regularization_coefficients))
    test_reg_losses = torch.zeros(len(regularization_coefficients))
    dms = torch.zeros(len(regularization_coefficients))
    eos = torch.zeros(len(regularization_coefficients))
    
    # Run experiments for each regularization coefficient
    for i, regularization_coefficient in enumerate(regularization_coefficients):
        train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.15, random_state=42)
        # Split the dataset and create data loaders
        
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, sampler=data.SubsetRandomSampler(train_indices), num_workers=4)
        val_loader = data.DataLoader(train_dataset, batch_size=batch_size, sampler=data.SubsetRandomSampler(val_indices), num_workers=4)

        # Initialize the model, optimizer, and loss function
        if model_type == "MLPComplex":
            #logging.info("MLP_model: {}".format(bert_model))
            model = MLPComplex(input_size = in_dim, output_size = out_dim, hidden_sizes = (100, 30, 10))
        elif model_type == "MLP":
            model = MLP(input_dim = in_dim, output_dim = out_dim, hidden_size = 10)
        else:
            model = LogisticRegression(input_dim = in_dim, output_dim = out_dim)
        model.to(device)
        if optimizer_str == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
        elif optimizer_str == "AdamW":
            optimizer = AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_str == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        #elif optimizer_str == "SparseAdam":
        #    optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate)
        else:
            print("CANNOT FIND OPTIMIZER", flush = True)

        num_classes = 2  # Number of sentiment classes (e.g., positive and negative)
        logging.info("Creating Model...")
        logging.info("Number of classes: {}".format(num_classes))
        logging.info("Model type: {}".format(model_type))
        logging.info("Data set type: {}".format(dataset))
        logging.info("Optimizer: {}".format(optimizer_str))


        train_size = len(train_loader.sampler)
        train_steps_per_epoch = train_size // batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=epochs * train_steps_per_epoch)

        criterion = nn.CrossEntropyLoss()#nn.MSELoss()#nn.BCELoss()#

        # Create the data loaders for the test and perturbed test sets
        test_dataset = COMPASDataset(simple_dataset, False, sensitive_feature)
        
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # Train the model with the current regularization coefficient
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            # Train for one epoch
            model.train()
            train_loss_avg = RunningAverage()

            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)")
            for batch in train_loader_tqdm:
                input_ids, labels, Z = batch
                input_ids, labels, Z = input_ids.to(device), labels.to(device), Z.to(device)

                logits, preds = model(input_ids)

                labels = torch.tensor(labels, dtype=torch.long)

                loss = criterion(logits, labels)
                
                # Compute the marginal regularization term
                mmd_loss = MMDLoss(kernel_bandwidth=None)
                mmd_loss.to(device)
                reg_term = compute_regularization_term(model, mmd_loss, logits, Z, labels, is_marginal_reg)     
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
            reg_loss = 0
            val_preds = []
            val_labels = []
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)")
            val_loss_avg = RunningAverage()
            with torch.no_grad():
                for batch in val_loader_tqdm:
                    input_ids, labels, Z = batch
                    input_ids, labels, Z = input_ids.to(device), labels.to(device), Z.to(device)

                    logits, preds = model(input_ids)
                    labels_long = torch.tensor(labels, dtype=torch.long)

                    loss = criterion(logits, labels_long)
                    val_loss += loss.item()

                    mmd_loss = MMDLoss(kernel_bandwidth=None)
                    mmd_loss.to(device)
                    reg_loss += compute_regularization_term(model, mmd_loss, logits, Z, labels, is_marginal_reg)  

                    val_loss_avg.update(loss.item())
                    # log the loss
                    val_loader_tqdm.set_postfix(loss='{:05.3f}'.format(val_loss_avg()))
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # Early stopping based on validation loss
            reg_loss /= len(val_loader)
            val_loss /= len(val_loader)
            in_domain_accuracy = np.mean(np.array(val_labels).flatten() == np.array(val_preds).flatten())
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

        # Evaluate the model on the original test set
        model.eval()
        test_acc_loss = 0
        test_reg_loss = 0
        test_preds = []
        test_labels = []
        test_logits = []
        test_zs = []
        test_pooling_outputs = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, labels, z_test = batch
                input_ids, labels, z_test = input_ids.to(device), labels.to(device), z_test.to(device)
                logits, preds = model(input_ids)

                labels_long = torch.tensor(labels, dtype=torch.long)
                loss = criterion(logits, labels_long)
                test_acc_loss += loss.item()

                mmd_loss = MMDLoss(kernel_bandwidth=None)
                mmd_loss.to(device)
                test_reg_loss += compute_regularization_term(model, mmd_loss, logits, z_test, labels, is_marginal_reg)  

                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_logits.extend(logits.cpu().numpy())
                test_zs.extend(z_test.cpu().numpy())
            test_reg_loss /= len(test_loader)
            test_acc_loss /= len(test_loader)
       
        in_domain_accuracy = np.mean(np.array(test_labels).flatten() == np.array(test_preds).flatten())
        test_accs[i] = in_domain_accuracy
        test_acc_losses[i] = test_acc_loss
        test_reg_losses[i] = test_reg_loss
        test_zs_tensor = torch.tensor(test_zs)
        test_preds_tensor = torch.tensor(test_preds)
        test_labels_tensor = torch.tensor(test_labels)
        
        # Evaluate the fairness metrics on the predictor
        dms[i] = fm.demographic_parity(test_preds_tensor, test_zs_tensor).item()
        eos[i] = fm.equalized_odds(test_labels_tensor, test_preds_tensor, test_zs_tensor)
        
        # Report the results
        logging.info(f"Regularization Coefficient: {regularization_coefficient}")
        logging.info(f"In-Domain Accuracy: {in_domain_accuracy}")
        logging.info(f"Final Demographic Parity: {dms[-1]}")
        logging.info(f"Final Equalized Odds: {eos[-1]}")
        logging.info("------------------------------------------------")
    printer = True
    if printer:
        print("STATS:")
        print("original data demographic parity: ", fm.demographic_parity(test_labels_tensor, test_zs_tensor).item())
        for i, rc in enumerate(regularization_coefficients):
            print("for coefficient ", rc, ":")
            print("accuracy: ", test_accs[i].item())
            print("demographic parity: ", dms[i].item())
            print("equalized odds: ", eos[i].item())
    return test_accs, test_acc_losses, test_reg_losses, dms, eos

def trials(n, rc):
  acc_c = torch.zeros(len(rc))
  acc_loss_c = torch.zeros(len(rc))
  reg_loss_c = torch.zeros(len(rc))
  dm_c = torch.zeros(len(rc))
  eo_c = torch.zeros(len(rc))
  acc_ac = torch.zeros(len(rc))
  acc_loss_ac = torch.zeros(len(rc))
  reg_loss_ac = torch.zeros(len(rc))
  dm_ac = torch.zeros(len(rc))
  eo_ac = torch.zeros(len(rc))
  for i in range(n):
    temp_a, temp_al, temp_rl, temp_dm, temp_eo = train(rc, True)
    acc_c += temp_a/n
    reg_loss_c += temp_rl/n
    acc_loss_c += temp_al/n
    dm_c += temp_dm/n
    eo_c +=  temp_eo/n
    temp_a, temp_al, temp_rl, temp_dm, temp_eo = train(rc, False)
    acc_ac += temp_a/n
    acc_loss_ac += temp_al/n
    reg_loss_ac += temp_rl/n
    dm_ac += temp_dm/n
    eo_ac +=  temp_eo/n
  return acc_c, acc_loss_c, reg_loss_c, dm_c, eo_c, acc_ac, acc_loss_ac, reg_loss_ac, dm_ac, eo_ac

def grapher(rc, acc, acc_loss, reg_loss, dm, eo, is_marginal_reg):
  if is_marginal_reg:
    title = "Causal"
    file_name = "causal"
  else:
    title = "Anti-causal"
    file_name = "anti-causal"
  #ACC AND LOSS PLOTS
  pyplot.plot(rc, acc, color = 'green', marker='.', label='Val Accuracy')
  pyplot.plot(rc, acc_loss, color = 'blue', marker='.', label='Accuracy Loss')
  pyplot.plot(rc, torch.tensor(rc)*reg_loss, color = 'cyan', marker='.', label='MMD Loss')
  pyplot.plot(rc, acc_loss + torch.tensor(rc)*reg_loss, marker='.', linestyle='--', color = 'black', label="Total Loss")
  # axis labels
  pyplot.xlabel('Regularizer Coefficient')
  pyplot.ylabel('Test Loss and Accuracy')
  # show the legend
  pyplot.legend()
  # add a title
  pyplot.title("Loss and Accuracy for " + title + " Regularizer")
  
  # save the plot
  pyplot.savefig(pre+'graphs/'+file_name+'_loss_acc.png')

  # show the plot
  pyplot.show()
  
  #FAIRNESS PLOTS
  pyplot.plot(rc, acc, color = 'green', marker='.',label='Val Accuracy')
  pyplot.plot(rc, dm, color = 'blue', marker='.', label='Demographic Parity')
  pyplot.plot(rc, eo, color = 'magenta', marker='.', label='Equalized Odds')
  pyplot.plot(rc, torch.zeros_like(eo), linestyle='--', color = 'black', label="Fairness")
  # axis labels
  pyplot.xlabel('Regularizer Coefficient')
  pyplot.ylabel('Various Measures')
  # show the legend
  pyplot.legend()
  # add a title
  if is_marginal_reg:
    pyplot.title("Fairness metric and Accuracy for Causal Regularizer")
  else:
    pyplot.title("Fairness metric and Accuracy for Anti-causal Regularizer")
  # save the plot
  pyplot.savefig(pre+'graphs/'+file_name+'_fairness_acc.png')
  #drive.download('/content/gdrive/MyDrive/MILA/Dhanya Stuff/course project/github/graphs/fairness_acc.png')
  # show the plot
  pyplot.show()

dataset = 'compas_simple'
batch_size = 256
learning_rate = 1e-3
patience = 10
num_tpus = 1
epochs = 10
kernel_bandwidth = 10
num_labels = 2
#is_marginal_reg = True
n_trials = 5
# TRIAL COEF
rc = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5,4, 4.5, 5, 6, 7.5, 9, 
      10, 15, 20, 25, 30, 40, 50, 75, 100, 250, 500]#, 750, 1000]
rc_dense = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5,4, 4.5, 5, 6, 7.5, 9, 
      10, 15, 20, 25, 30, 40, 50, 75, 100]
rc_1 = [112, 125, 137, 150, 162, 175, 187, 200, 212, 225, 237, 275, 300, 400]
rc_2 = [12.5, 17.5, 22.5, 27.5, 32.5, 35, 37.5, 42.5, 45, 47.5, 55, 60,
        70, 80, 85, 90, 95]
rc_3 = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 
        5.25, 5.5, 5.75, 6.25, 6.5, 6.75, 7, 8, 8.5]
rc_4 = [325, 350, 375, 425, 450, 475]
rc_test = [0, 0.5, 1]


acc_c, acc_loss_c, reg_loss_c, dm_c, eo_c, acc_ac, acc_loss_ac, reg_loss_ac, dm_ac, eo_ac = trials(n_trials, rc_test)

stats_dict = {
    'regularization_coefficients': rc_test,
    'causal_accuracy': acc_c,
    'causal_accuracy_loss': acc_loss_c,
    'causal_MMD_loss': reg_loss_c,
    'causal_demographic_parity': dm_c,
    'causal_equalized_odds': eo_c,
    'anti_causal_accuracy': acc_ac,
    'anti_causal_accuracy_loss': acc_loss_ac,
    'anti_causal_MMD_loss': reg_loss_ac,
    'anti_causal_demographic_parity': dm_ac,
    'anti_causal_equalized_odds': eo_ac
}

# Save the dictionary to a CSV file
df = pd.DataFrame(stats_dict)
df.to_csv(pre+'graphs/data_test.csv', index=False)

#grapher(rc_4,  acc_c, acc_loss_c, reg_loss_c, dm_c, eo_c, True)
#grapher(rc_4,  acc_ac, acc_loss_ac, reg_loss_ac, dm_ac, eo_ac, False)

#grapher(rc_dense,  acc_c[:-4], acc_loss_c[:-4], reg_loss_c[:-4], dm_c[:-4], eo_c[:-4], True)
#grapher(rc_dense,  acc_ac[:-4], acc_loss_ac[:-4], reg_loss_ac[:-4], dm_ac[:-4], eo_ac[:-4], False)