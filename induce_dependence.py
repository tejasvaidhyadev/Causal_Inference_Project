import numpy as np
import pandas as pd

# Boilerplate implementation of inducing dependence in a dataset for real world data 
# ToDo - Implement this in gen_dataset.py

def induce_dependence(data, gamma):
    # Step 1: Randomly drop reviews with 0 helpful votes V
    data = data[data['helpful_votes'] > 0].copy()

    while True:
        z1 = data[data['Z'] == 1]
        z0 = data[data['Z'] == 0]
        p_z1 = len(z1[z1['helpful_votes'] > 0]) / len(z1)
        p_z0 = len(z0[z0['helpful_votes'] > 0]) / len(z0)

        if p_z1 > gamma and p_z0 > (1 - gamma):
            break
        data = data.sample(frac=0.9).reset_index(drop=True)

    # Step 2: Find the smallest Tz
    T1 = 0
    T0 = 0
    while True:
        p_v_t1_z1 = len(z1[z1['helpful_votes'] > T1]) / len(z1)
        p_v_t0_z0 = len(z0[z0['helpful_votes'] > T0]) / len(z0)

        if p_v_t1_z1 < gamma and p_v_t0_z0 < (1 - gamma):
            break
        T1 += 1
        T0 += 1

    # Step 3: Set Y
    data['Y'] = 0
    data.loc[data['Z'] == 0, 'Y'] = (data.loc[data['Z'] == 0, 'helpful_votes'] > T0).astype(int)
    data.loc[data['Z'] == 1, 'Y'] = (data.loc[data['Z'] == 1, 'helpful_votes'] > T1).astype(int)
    
    return data
def test_data():
    data = pd.DataFrame({
        'Z': [0, 0, 0, 1, 1, 1],
        'helpful_votes': [0, 1, 2, 0, 1, 2]
    })
    return data

data = test_data()
# Load your dataset into a DataFrame and replace 'data' below with the loaded DataFrame
# data = pd.DataFrame()
gamma = 0.5
induced_data = induce_dependence(data, gamma)
print(induced_data)
