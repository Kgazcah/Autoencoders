import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from preprocessing.interface_builder import Builder
from preprocessing.preprocessor_builder import Preprocessing
from preprocessing.director import Director
from vocabulary.getting_vocabulary import GettingVocabulary
from encoding.lambda_grams import LambdaGrams
from encoding.lambda_grams_to_indx import LambdaGramsToIndx
from encoding.binary_embeddings import LambdaGramEmbeddings

df = pd.read_csv('Data/U2_EXER_W2V.csv')

####################### Step 1: Preprocessing (Builder Pattern)
director = Director()
builder = Preprocessing(df)
corpus_stopwords = director.makeBasicPreprocessing(builder) #basic con stopwords
corpus_no_stopwords = director.makePlusPreprocessing(builder) #plus sin stopwords
# print(corpus_stopwords)
# print(corpus_no_stopwords)

###################### Step 2: Getting the vocabulary and their indexes

vocab_obj = GettingVocabulary(corpus_stopwords)
vocabulary = vocab_obj.get_vocab()
print("SIIIIIIIIIIN STOPWORDS")
print(type(vocabulary)) #with stopwords 2093 without stopwords 1986


vocab_to_index = vocab_obj.get_vocab_to_indx()
print(vocab_to_index)

word_to_bin, bin_to_word, bits = vocab_obj.get_binary_rep(vocab_to_index)
print(word_to_bin)
# print(bin_to_word)
print(bits)

##################### Step 3: Getting lambda grams

n_grams = LambdaGrams(corpus_stopwords)
lambda_grams_3 = n_grams.get_lambda_grams(3,1)
print(lambda_grams_3) #len 8941

lambda_grams_4 = n_grams.get_lambda_grams(4,1)
print(len(lambda_grams_4)) #len 2171

lambda_grams_5 = n_grams.get_lambda_grams(5,1)
print(len(lambda_grams_5)) #len 2158


##################### Step 4: Lambda grams to indexes 
# print(vocab_to_index)
mapping = LambdaGramsToIndx(lambda_grams_3, vocab_to_index)
result = mapping.get_lambda_grams_indx()
# print(result)

##################### Step 5: Encoding the lambda grams

binary_encode = LambdaGramEmbeddings(word_to_bin)
binary_embeddings_3 = binary_encode.get_embeddings_df(lambda_grams_3)
binary_embeddings_3.to_csv('assets/binary_embeddings_3.csv', index=False)

binary_embeddings_4 = binary_encode.get_embeddings_df(lambda_grams_4)
binary_embeddings_4.to_csv('assets/binary_embeddings_4.csv', index=False)

binary_embeddings_5 = binary_encode.get_embeddings_df(lambda_grams_5)
binary_embeddings_5.to_csv('assets/binary_embeddings_5.csv', index=False)






