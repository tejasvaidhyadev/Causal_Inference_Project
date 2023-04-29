# This implementation is from "Robustness to Stress Tests synthetic data"
# This looks like potentially a good way to induce dependence in a dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(reviews, n_samples=18000): # 18000
    reviews_subset = reviews.sample(n=n_samples, random_state=42)
    Y = np.where(reviews_subset['score'].isin([4, 5]), 1, 0)
    X = reviews_subset['text'].str.split().str[:20].str.join(' ')
    
    return X, Y

def assign_Z_variable(X):
    Z = np.random.binomial(1, 0.5, size=len(X))
    return Z

def replace_tokens(X, Z):
    X = np.where(Z == 1, X.str.replace('and', 'andxxxxx').str.replace('the', 'thexxxxx'), 
                 X.str.replace('and', 'andyyyyy').str.replace('the', 'theyyyyy'))
    return X

def induce_association( Y, Z, gamma=0.3, p_Y=0.5):
    # Calculate the number of samples required for each group
    n_z1_y1 = int(sum((Z == 1) & (Y == 1)) * gamma / (1 - gamma))
    n_z0_y0 = int(sum((Z == 0) & (Y == 0)) * gamma / (1 - gamma))

    # Randomly sample the required number of samples from each group
    z1_y1_indices = np.random.choice(np.where((Z == 1) & (Y == 1))[0], size=n_z1_y1, replace=False)
    z0_y0_indices = np.random.choice(np.where((Z == 0) & (Y == 0))[0], size=n_z0_y0, replace=False)

    # Update Y values to create the target association
    print(len(Y))
    print(len(z1_y1_indices))
    print(len(z0_y0_indices))
    Y[z1_y1_indices] = 0
    Y[z0_y0_indices] = 1
    return  Y


def split_data(X, Y, Z, test_size=0.2):
    return train_test_split(X, Y, Z, test_size=test_size, random_state=42)

def save_data_to_csv(X, Y, Z, file_path):
    if not os.path.exists('data'):
        os.makedirs('data')
        
    pd.DataFrame({'X': X, 'Y': Y, 'Z': Z}).to_csv(file_path, index=False)

def create_perturbed_dataset(test_df):
    perturbed_df = test_df.copy()
    perturbed_df['X'] = perturbed_df['X'].str.replace('yyyyy', '').str.replace('xxxxx', '')
    perturbed_df['X'] = np.where(perturbed_df['Z'] == 1, 
                                  perturbed_df['X'].str.replace('and', 'andyyyyy').str.replace('the', 'theyyyyy'),
                                  perturbed_df['X'].str.replace('and', 'andxxxxx').str.replace('the', 'thexxxxx'))
    return perturbed_df

if __name__ == "__main__":
    reviews = load_data('data/amazon_reviews.csv')
    X, Y = preprocess_data(reviews)
    # generate Z variable using a Bernoulli distribution
    Z = assign_Z_variable(X)

    # replace tokens in X based on Z variable based on the paper
    X = replace_tokens(X, Z)

    # induce association between Y and Z
    X_resampled, Y_resampled, Z_resampled = X, induce_association(Y, Z), Z
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = split_data(X_resampled, Y_resampled, Z_resampled)
    
    save_data_to_csv(X_train, Y_train, Z_train, 'data/train.csv')
    save_data_to_csv(X_test, Y_test, Z_test, 'data/test.csv')
    
    test_df = load_data('data/test.csv')
    perturbed_df = create_perturbed_dataset(test_df)
    perturbed_df.to_csv('data/perturbed_test.csv', index=False)
