import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen

# This codebase isn't meant to be run as a script, but if you want to download the data yourself, you can uncomment the following lines

# Use this in your code
# !wget http://deepyeti.ucsd.edu/jianmo/amazon/sample/meta_Computers.json.gz

### load the meta data

data = []
with gzip.open('meta_Computers.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))
    
# total length of list, this number equals total number of products
print(len(data))

# first row of the list
print(data[0])


df = pd.DataFrame.from_dict(data)

print(len(df))
df3 = df.fillna('')
df4 = df3[df3.title.str.contains('getTime')] # unformatted rows
df5 = df3[~df3.title.str.contains('getTime')] # filter those unformatted rows
print(len(df4))
print(len(df5))