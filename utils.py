from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from autoencoder.nn import Autoencoder
from preprocessing.interface_builder import Builder
from preprocessing.director import Director
from preprocessing.preprocessor_builder import Preprocessing

#split the original dataset to train
def split_data (df, input_folder, output_folder, test_size=0.20, random_state=42):
    df = pd.read_csv(f'{input_folder}/{df}')
    X_train, X_test, y_train, y_test = train_test_split(
    df, df, test_size=test_size, random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42)

    #as 'y' data must be the same as 'X' data we do not save it in all cases
    X_train = pd.DataFrame(X_train)
    X_train.to_csv(f'{output_folder}/X_train.csv', index=False)

    X_test = pd.DataFrame(X_test)
    X_test.to_csv(f'{output_folder}/X_test.csv', index=False)

    X_val = pd.DataFrame(X_val)
    X_val.to_csv(f'{output_folder}/X_val.csv', index=False)

#preprocessing the corpus (Builder Pattern)
def preprocessing(df, type):
    director = Director()
    builder = Preprocessing(df)
    if type == 'basic':
        #basic builder includes stopwords
        preprocessed_df = director.makeBasicPreprocessing(builder)
    elif type == 'plus':
        #plus builder does not include stopwords
        preprocessed_df = director.makePlusPreprocessing(builder)
    return preprocessed_df

#to do: adding match_counts as a function
df = pd.read_csv('data/U2_EXER_W2V.csv')
df_p = preprocessing(df, 'basic')
print(df_p)