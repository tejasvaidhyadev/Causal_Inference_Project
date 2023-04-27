# This implementation is from "Robustness to Stress Tests synthetic data"
# This looks like potentially a good way to induce dependence in a dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(reviews, n_samples=18000):
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

def resample_data(X, Y, gamma=0.3, p_Y=0.5):
    X_resampled = np.random.permutation(X)
    Y_resampled = np.random.binomial(1, p_Y, size=len(Y))
    Z_resampled = np.where(Y_resampled == 1, np.random.binomial(1, gamma, size=len(Y)),
                              np.random.binomial(1, 1 - gamma, size=len(Y)))
    
    return X_resampled, Y_resampled, Z_resampled

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
    Z = assign_Z_variable(X)
    X = replace_tokens(X, Z)
    X_resampled, Y_resampled, Z_resampled = resample_data(X, Y)
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = split_data(X_resampled, Y_resampled, Z_resampled)
    
    save_data_to_csv(X_train, Y_train, Z_train, 'data/train.csv')
    save_data_to_csv(X_test, Y_test, Z_test, 'data/test.csv')
    
    test_df = load_data('data/test.csv')
    perturbed_df = create_perturbed_dataset(test_df)
    perturbed_df.to_csv('data/perturbed_test.csv', index=False)
