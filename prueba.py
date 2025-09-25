############################ Adding my embeddings to yuri's embeddings
import pandas as pd

problem = 'software_requirements/stopwords'

df = pd.read_csv(f'data/{problem}/dataset_ngrams.csv')

df_train = pd.read_csv(f'data/{problem}/train_df.csv')
df_test = pd.read_csv(f'data/{problem}/test_df.csv')
df_val = pd.read_csv(f'data/{problem}/val_df.csv')

cols_to_add = ['text', '1_gram_embeddings']

df_subset = df[cols_to_add]

df_train = df_train.merge(df_subset, on='text', how='left')
df_test  = df_test.merge(df_subset, on='text', how='left')
df_val   = df_val.merge(df_subset, on='text', how='left')

df_train.to_csv(f'data/{problem}/train_df.csv', index=False)
df_test.to_csv(f'data/{problem}/test_df.csv', index=False)
df_val.to_csv(f'data/{problem}/val_df.csv', index=False)