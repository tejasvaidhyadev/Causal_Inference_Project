# Causal Infernence course project
Unofficial repository for the the paper ["Counterfactual Invariance to Spurious Correlations: Why and How to Pass Stress Tests"](https://arxiv.org/pdf/2106.00545.pdf) 

# Generate Datasets
gen_dataset only supports generating synthetic datasets for now.  
Although I have added script for generating real datasets, it is not yet tested. checkout for ```induce_dependence.py``` for more details.

# Setup
- Clone this repository  
- Install the required dependencies (check env.yml) 

# Training
Added to version of trainer script. Use pytorch_trainer.py for training with pytorch and tensorflow_trainer.py for training with tensorflow.  
Note: pytorch_trainer.py is more feature rich and is recommended.  

```
python pytorch_trainer.py 

```

--dataset: Directory containing the dataset. Default value is synthetic_exp3.  
--batch_size: Batch size for training. Default value is 80.  
--learning_rate: Learning rate for training. Default value is 5e-5.  
--patience: Patience for early stopping. Default value is 5.  
--num_tpus: Number of TPUs to use. Default value is 1.  
--epochs: Number of epochs to train for. Default value is 1.  
--num_labels: Number of labels in the dataset. Default value is 2.  
--max_seq_length: Maximum sequence length for BERT. Default value is 128.  
--is_marginal_reg: Whether to use marginal regularization. Default value is True.  

