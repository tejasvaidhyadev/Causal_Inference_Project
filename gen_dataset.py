import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
# script try to replicate figure 2 from the paper
# Load the Amazon reviews data
reviews = pd.read_csv('data/amazon_reviews.csv')

# Select 20000 product reviews and assign labels based on the star rating
reviews_subset = reviews.sample(n=20000, random_state=42)
Y = np.where(reviews_subset['score'].isin([4, 5]), 1, 0)

# Use only the first 20 tokens of text
X = reviews_subset['text'].str.split().str[:20].str.join(' ')

# Assign Z as a Bernoulli random variable
Z = np.random.binomial(1, 0.5, size=len(X))

# Replace the tokens "and" and "the" based on the value of Z
X = np.where(Z == 1, X.str.replace('and', 'andxxxxx').str.replace('the', 'thexxxxx'), 
             X.str.replace('and', 'andyyyyy').str.replace('the', 'theyyyyy'))

# Resample the data to induce a dependency between Y and Z
gamma = 0.3
p_Y = 0.5
n_Y = int(len(Y) * p_Y)
n_Z = int(len(Z) * gamma)
idx_Y1Z1 = np.where((Y == 1) & (Z == 1))[0]
idx_Y1Z0 = np.where((Y == 1) & (Z == 0))[0]
idx_Y0Z1 = np.where((Y == 0) & (Z == 1))[0]
idx_Y0Z0 = np.where((Y == 0) & (Z == 0))[0]
idx_Y1Z1_resampled = np.random.choice(idx_Y1Z1, size=int(n_Y*n_Z*p_Y), replace=True)
idx_Y1Z0_resampled = np.random.choice(idx_Y1Z0, size=int(n_Y*(1-gamma)*p_Y), replace=True)
idx_Y0Z1_resampled = np.random.choice(idx_Y0Z1, size=int((1-p_Y)*n_Z*gamma), replace=True)
idx_Y0Z0_resampled = np.random.choice(idx_Y0Z0, size=int((1-p_Y)*(1-gamma)*n_Z), replace=True)
idx_resampled = np.concatenate([idx_Y1Z1_resampled, idx_Y1Z0_resampled, idx_Y0Z1_resampled, idx_Y0Z0_resampled])
X_resampled = X[idx_resampled]
Y_resampled = Y[idx_resampled]
Z_resampled = Z[idx_resampled]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X_resampled, Y_resampled, Z_resampled, 
                                                                    test_size=0.2, random_state=42)

# save the dataset to csv
# check and create data folder
import os
if not os.path.exists('data'):
    os.makedirs('data')

pd.DataFrame({'X': X_train, 'Y': Y_train, 'Z': Z_train}).to_csv('data/train.csv', index=False)
pd.DataFrame({'X': X_test, 'Y': Y_test, 'Z': Z_test}).to_csv('data/test.csv', index=False)
