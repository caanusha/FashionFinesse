import os
import pickle

from nomic import atlas
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

DATASET_PATH = "C:/Users/aishu/Downloads/fashion-dataset/"

df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=7103, on_bad_lines='skip')
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)
data = df[['usage','gender','articleType']].to_dict("records")
with open('fashion.pkl', 'rb') as f:
    fashion_embed = pickle.load(f) # deserialize using load()

project = atlas.map_embeddings(embeddings=np.array(fashion_embed), data=data)
