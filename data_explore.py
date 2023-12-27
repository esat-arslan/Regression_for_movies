# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:20:41 2023

@author: ars16
"""

import os
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

dataset= pd.read_csv('data_mining.csv')
print("first five row:")
print(dataset.head())
print("number of empty values:")
print(dataset.isnull().sum())


#dataset = dataset.dropna()

dataset['Genre'] = dataset['Genre'].fillna('')  

  

genres = [genre.split(",") for genre in dataset['Genre']]
flat_genres = [genre.strip() for sublist in genres for genre in sublist]
genre_counts = pd.Series(flat_genres).value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
plt.title('Distribution of Movies by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')
plt.show()
income_by_vote_average = dataset.groupby('Vote Average')['Income'].mean().reset_index()

plt.figure(figsize=(12, 8))
sns.lineplot(x='Vote Average', y='Income', data=income_by_vote_average, marker='o', color='blue')
plt.title('Average Income by Vote Average')
plt.xlabel('Vote Average')
plt.ylabel('Average Income')
plt.show()



bins = [0, 1, 2, 3, 4, 5,6,7,8,9,10]  
labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10']  
dataset['Vote Average Group'] = pd.cut(dataset['Vote Average'], bins=bins, labels=labels, include_lowest=True)

income_by_vote_average = dataset.groupby('Vote Average Group')['Income'].mean().reset_index()

plt.figure(figsize=(12, 8))
sns.barplot(x='Vote Average Group', y='Income', data=income_by_vote_average, color='blue')
plt.title('Average Income by Vote Average Group')
plt.xlabel('Vote Average Group')
plt.ylabel('Average Income')
plt.show()


genre_column = 'Genre'
dataset = dataset.dropna(subset=['Genre'])

all_genres = dataset[genre_column].str.split(',').explode().str.strip()

genre_income_df = pd.DataFrame({'Genre': all_genres, 'Income': dataset['Income']})
genre_mean_income = genre_income_df.groupby('Genre')['Income'].mean()


genres_df = dataset['Genre'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).to_frame('Genre')
df_split_genres = dataset.drop('Genre', axis=1).join(genres_df).reset_index(drop=True)



plt.figure(figsize=(12, 6))
genre_mean_income.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Average Income by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Income')
plt.xticks(rotation=45, ha='right')
plt.show()

genres_df = dataset['Genre'].str.split(',').explode().str.strip()

df_split_genres = pd.DataFrame({'Genre': genres_df, 'Vote Average': dataset['Vote Average']})

plt.figure(figsize=(14, 6))
sns.boxplot(x='Genre', y='Vote Average', data=df_split_genres)
plt.title('Relationship Between Genres and Vote Average')
plt.xticks(rotation=45, ha='right')
plt.show()