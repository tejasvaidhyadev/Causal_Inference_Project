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

def induce_association( data , gamma=0.3, p_y_1=0.5):
    
    num_samples = len(data)
    
    y_equals_1 = int(num_samples * p_y_1)
    y_equals_0 = num_samples - y_equals_1

    z_1_y_1 = int(y_equals_1 * gamma)
    z_0_y_1 = y_equals_1 - z_1_y_1
    
    z_1_y_0 = int(y_equals_0 * (1 - gamma))
    z_0_y_0 = y_equals_0 - z_1_y_0

    anti_causal_data = pd.concat([
        data[(data['Y'] == 1) & (data['Z'] == 1)].sample(n=z_1_y_1, replace=True),
        data[(data['Y'] == 1) & (data['Z'] == 0)].sample(n=z_0_y_1, replace=True),
        data[(data['Y'] == 0) & (data['Z'] == 1)].sample(n=z_1_y_0, replace=True),
        data[(data['Y'] == 0) & (data['Z'] == 0)].sample(n=z_0_y_0, replace=True)
    ])

    return  anti_causal_data


def split_data(data, test_size=0.2):
    return train_test_split(data, test_size=test_size, random_state=42)

def save_data_to_csv(data, file_path):
    if not os.path.exists('data'):
        os.makedirs('data')
    data.to_csv(file_path, index=False)

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
    
    # make new data frame
    data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    
    
    # INDUCE Assoication
    anti_causal_data = induce_association( data , gamma=0.3, p_y_1=0.5)
    
    # Split the data into train (70%), test (30%) sets
    train_data, test_data = split_data(anti_causal_data, test_size=0.2)

    #train_data, test_data = train_test_split(anti_causal_data, test_size=43783 + 17513, random_state=42)

    # induce association between Y and Z    
    save_data_to_csv(train_data, 'data/train.csv')
    save_data_to_csv(test_data, 'data/test.csv')
    #train_data.to_csv('data/train.csv', index=False)
    #test_data.to_csv('data/test.csv', index=False)

    test_df = load_data('data/test.csv')
    perturbed_df = create_perturbed_dataset(test_df)
    perturbed_df.to_csv('data/perturbed_test.csv', index=False)
    print("I am done!")
