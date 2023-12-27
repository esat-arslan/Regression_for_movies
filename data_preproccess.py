# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models,matutils
import seaborn as sns
import matplotlib.pyplot as plt


dataset_path = "data_minning.csv"  
df = pd.read_csv(dataset_path)
df['Genre'].fillna('', inplace=True)
#dataset = dataset.dropna()
#Extract release date and month to two seperate columns
df['Release Date'] = pd.to_datetime(df['Release Date'])
df['Month'] = df['Release Date'].dt.month
df['Day'] = df['Release Date'].dt.day
#one hot encoding for Genre data
genres_list = df['Genre'].str.split(',').explode().str.strip()

unique_genres = genres_list.unique()

for genre in unique_genres:
    df[genre] = df['Genre'].str.contains(genre, case=False, regex=False).astype(int)

df.drop('Genre', axis=1, inplace=True)
df = df[df[unique_genres].sum(axis=1) > 0]

# Normalize 'Popularity', 'Vote Average', and 'Vote Count'
columns_to_normalize = ['Popularity', 'Vote Average', 'Vote Count']
min_max_scaler = MinMaxScaler()
df[columns_to_normalize] = min_max_scaler.fit_transform(df[columns_to_normalize])

preprocessed_dataset_path = "preprocessed__.csv"  
df.to_csv(preprocessed_dataset_path, index=False)

print(f"\nPreprocessed dataset is saved to: {preprocessed_dataset_path}")