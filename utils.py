from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ast
from autoencoder.nn import Autoencoder
from preprocessing.interface_builder import Builder
from preprocessing.director import Director
from preprocessing.preprocessor_builder import Preprocessing
from vocabulary.getting_vocabulary import GettingVocabulary
from encoding.lambda_grams import LambdaGrams
from encoding.lambda_grams_to_indx import LambdaGramsToIndx
from encoding.binary_embeddings import LambdaGramEmbeddings

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



#Getting the vocabulary and their indexes
def get_vocab_ind_bin(preprocessed_df, output_file='assets/method'):
    vocab_obj = GettingVocabulary(preprocessed_df)
    # get vocabulary (unique words) with stopwords 2093 and without stopwords 1986
    vocabulary = vocab_obj.get_vocab()
    vocab = pd.DataFrame(vocabulary)
    vocab.to_csv(f'{output_file}/vocabulary.csv', index=False)

    #assign a decimal index to each word from the vocabulary
    #vocabulary to index document will have something as follows:
    #{'x': 1, ..., 'yet': 1984, 'zero': 1985}
    vocab_to_index = vocab_obj.get_vocab_to_indx()
    vocab_to_index_df = pd.DataFrame(list(vocab_to_index.items()), 
                                    columns=['word', 'index'])
    vocab_to_index_df.to_csv(f'{output_file}/vocab_to_index.csv', index=False)

    # If you want to upload your own vocabulary to index, uncomment the next 3 lines.
    # The dictionary must be in the form: #{'x': 1, ..., 'yet': 1984, 'zero': 1985}
    # vocab_to_index_df = pd.read_csv('assets/vocab_to_index.csv')
    # columns = ['word', 'index']
    # vocab_to_index = vocab_to_index_df.set_index(columns[0])[columns[1]].to_dict()

    #get the binary embedding from the decimal indexes
    vocab_to_binary = vocab_obj.get_binary_rep(vocab_to_index)

    #the embedding binary dictionary has the following example form:
    # {'yet': '110000',..., 'zero': '100001'}
    binary_dic = pd.DataFrame(list(vocab_to_binary.items()), columns=['word', 'binary'])
    binary_dic.to_csv(f'{output_file}/vocab_to_binary.csv', index=False)
    return vocab_to_binary, vocab_to_index

#Getting lambda grams
def get_lambda_grams(preprocessed_df, n_gram):
    n_grams = LambdaGrams(preprocessed_df)
    #for 3 gram example we will have something as follows: 
    #  ['carried distributed manner']
    lambda_grams = n_grams.get_lambda_grams(n_gram)
    return lambda_grams

# Lambda grams to indexes
def lambda_grams_to_indexes(lambda_grams, vocab_to_index):
    #convert the lambda grams to indexes (e.g., [259, 533, 1060] )
    #for this we need the vocabulary to index document of the form:
    #{'x': 1, ..., 'yet': 1984, 'zero': 1985}

    lti = LambdaGramsToIndx(lambda_grams, vocab_to_index)
    lambda_gram_to_index = lti.get_lambda_grams_indx()


#Encoding the lambda grams
def lambda_grams_to_binary(vocab_to_binary, lambda_grams, output_file_name):
    binary_encode_decode = LambdaGramEmbeddings(vocab_to_binary)
    dictionary = vocab_to_binary
    #saving the binary embeddings
    binary_lambda_grams = binary_encode_decode.get_embeddings_df(lambda_grams)
    binary_lambda_grams.to_csv(output_file_name, index=False)
    return dictionary

def binary_to_ngrams(binary_embedding, ind, n_gram, vocab_to_binary):
    binary_encode_decode = LambdaGramEmbeddings(vocab_to_binary)
    return binary_encode_decode.binary_to_ngrams(binary_embedding[ind], n_gram, vocab_to_binary)

#Uploading the data
def upload_data_to_train(file_name, column):
    df = pd.read_csv(file_name)
    return np.vstack(df[column].apply(lambda x: np.array(list(x), dtype=int)).values)


def upload_vocab_to_binary_dictionary(file='binary_dict_karina.csv', columns=['word', 'binary']):
    # The dictionary must be in the form: {'yet': '110000', 'zero': '100001'}
    df = pd.read_csv(file, dtype={'binary': str})
    dictionary = df.set_index(columns[0])[columns[1]].to_dict()
    return dictionary

# to do: adding match_counts as a function


 




