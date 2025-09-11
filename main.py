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
from autoencoder.nn import Autoencoder
from visualization.plotting import Visualization

df = pd.read_csv('Data/U2_EXER_W2V.csv')

####################### Step 1: Preprocessing (Builder Pattern)
director = Director()
builder = Preprocessing(df)
#basic builder includes stopwords
corpus_stopwords = director.makeBasicPreprocessing(builder)
#plus builder does not include stopwords
corpus_no_stopwords = director.makePlusPreprocessing(builder) 

###################### Step 2: Getting the vocabulary and their indexes
vocab_obj = GettingVocabulary(corpus_no_stopwords)
# get vocabulary (unique words) with stopwords 2093 and without stopwords 1986
vocabulary = vocab_obj.get_vocab()
vocab = pd.DataFrame(vocabulary)
vocab.to_csv('assets/method/1_vocabulary.csv', index=False)

#assign a decimal index to each word from the vocabulary
#vocabulary to index document will have something as follows:
#{'x': 1, ..., 'yet': 1984, 'zero': 1985}
vocab_to_index = vocab_obj.get_vocab_to_indx()
vocab_to_index_df = pd.DataFrame(list(vocab_to_index.items()), 
                                 columns=['word', 'index'])
vocab_to_index_df.to_csv("assets/method/2_vocab_to_index.csv", index=False)

# If you want to upload your own vocabulary to index, uncomment the next 3 lines.
# The dictionary must be in the form: #{'x': 1, ..., 'yet': 1984, 'zero': 1985}
# vocab_to_index_df = pd.read_csv('assets/vocab_to_index.csv')
# columns = ['word', 'index']
# vocab_to_index = vocab_to_index_df.set_index(columns[0])[columns[1]].to_dict()

#get the binary embedding from the decimal indexes
word_to_bin, bin_to_word, bits = vocab_obj.get_binary_rep(vocab_to_index)

#the embedding binary dictionary has the following example form:
# {'yet': '110000',..., 'zero': '100001'}
binary_dic = pd.DataFrame(list(word_to_bin.items()), columns=['word', 'binary'])
binary_dic.to_csv('assets/method/3_binary_dict.csv', index=False)

##################### Step 3: Getting lambda grams

n_grams = LambdaGrams(corpus_no_stopwords)
#for 3 gram example we will have something as follows: 
#  ['carried distributed manner']
lambda_grams_3 = n_grams.get_lambda_grams(3,1) #len 8941
lambda_grams_4 = n_grams.get_lambda_grams(4,1) #len 2171
lambda_grams_5 = n_grams.get_lambda_grams(5,1) #len 2158


##################### Step 4: Lambda grams to indexes 
#convert the lambda grams to indexes (e.g., [259, 533, 1060] )
#for this we need the vocabulary to index document of the form:
#{'x': 1, ..., 'yet': 1984, 'zero': 1985}

lti = LambdaGramsToIndx(lambda_grams_3, vocab_to_index)
lambda_gram_to_index = lti.get_lambda_grams_indx()

##################### Step 5: Encoding the lambda grams

binary_encode_decode = LambdaGramEmbeddings(word_to_bin)
dictionary = word_to_bin

# If you want to upload your own dictionary, uncomment the next 3 lines.  
# The dictionary must be in the form: {'yet': '110000', 'zero': '100001'}
# dictionary_df = pd.read_csv('binary_dict_karina.csv', dtype={"binary": str})
# columns = ['word', 'binary']
# dictionary = dictionary_df.set_index(columns[0])[columns[1]].to_dict()

#to decode a binary string, uncomment the next 3 lines
# binary_embedding_test = '111001001001100101111011011101110'
# n_grams_in_a_binary = binary_encode_decode.binary_to_ngrams(binary_embedding_test, 3, dictionary)
# print(f"The binary embedding {binary_embedding_test} contains the tokens {n_grams_in_a_binary}.")

#saving the binary embeddings
binary_embeddings_3 = binary_encode_decode.get_embeddings_df(lambda_grams_3)
binary_embeddings_3.to_csv('assets/bin_embeddings/binary_embeddings_3.csv', index=False)

binary_embeddings_4 = binary_encode_decode.get_embeddings_df(lambda_grams_4)
binary_embeddings_4.to_csv('assets/bin_embeddings/binary_embeddings_4.csv', index=False)

binary_embeddings_5 = binary_encode_decode.get_embeddings_df(lambda_grams_5)
binary_embeddings_5.to_csv('assets/bin_embeddings/binary_embeddings_5.csv', index=False)

##################### Step 6: Splitting the data
# from sklearn.model_selection import train_test_split

# df_b = pd.read_csv('assets/bin_embeddings/binary_embeddings_3.csv')

# X_train, X_test, y_train, y_test = train_test_split(
# df_b, df_b, test_size=0.20, random_state=42)

# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.20, random_state=42)

# X_train = pd.DataFrame(X_train)
# X_train.to_csv("data/X_train.csv", index=False)
# y_train = pd.DataFrame(y_train)
# y_train.to_csv("data/y_train.csv", index=False)

# X_test = pd.DataFrame(X_test)
# X_test.to_csv("data/X_test.csv", index=False)
# y_test = pd.DataFrame(y_test)
# y_test.to_csv("data/y_test.csv", index=False)

# X_val = pd.DataFrame(X_val)
# X_val.to_csv("data/X_val.csv", index=False)
# y_val = pd.DataFrame(y_val)
# y_val.to_csv("data/y_val.csv", index=False)

####################### Step 7: Uploading the data

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
X_val = pd.read_csv('data/X_val.csv')

X_train = np.vstack(X_train['embedding'].apply(lambda x: np.array(list(x), dtype=int)).values)
X_test = np.vstack(X_test['embedding'].apply(lambda x: np.array(list(x), dtype=int)).values)
X_val = np.vstack(X_val['embedding'].apply(lambda x: np.array(list(x), dtype=int)).values)
y_train = X_train
y_test = X_test
y_val = X_val

##################### Step 8: Creating and training the Neural Network (Autoencoder)
'''
autoencoder = Autoencoder(X_train.shape[1], y_train.shape[1])
history = autoencoder.fit(X_train, y_train, X_val, y_val, epochs=200, batch_size=32)

autoencoder.save()

##################### Step 9: Visualizing the training plots
plot = Visualization()
plot.plotting_metric(history.history, 'cosine_similarity', 'val_cosine_similarity', path='assets/learning_graphs', fig_name='Learning training')
plot.plotting_loss(history.history, 'loss', 'val_loss', path='assets/learning_graphs', fig_name='Loss training')
'''
#################### Step 10: Predicting
#comment following line if you want to train
autoencoder = Autoencoder(X_train.shape[1], y_train.shape[1])
model = autoencoder.load_model()
y_pred = model.predict(X_test)
y_pred = y_pred.round(0)
print(y_pred[0])
print(X_test[0])

#################### Step 11: Encode

encode = autoencoder.encode()
n_grams_embeddings = encode.predict(X_test)
df_embeddings = pd.DataFrame(n_grams_embeddings)
df_embeddings.to_csv('assets/n_gram_embeddings/n_gram_embedding.tsv', sep="\t", index=False, header=False)

n_grams_df = pd.read_csv('data/X_test.csv')
n_grams = n_grams_df['lambda_gram']
n_grams.to_csv('assets/n_gram_embeddings/n_gram_words.tsv', sep="\t", index=False, header=False)


##################### Step 12: Decode
decode = autoencoder.decode()
bin_embedding = decode.predict(n_grams_embeddings)
bin_embedding = np.round(bin_embedding, 0)
print(f'Original Embedding: {n_grams_embeddings[2]}')
print(f'Predicted Binary embedding: {bin_embedding[2]}')
original_binary_embedding = n_grams_df['embedding'][2]
print(f'Original Binary embedding: {original_binary_embedding}')
print(f'Original N-grams: {n_grams[2]}')
print(f'Predicted N-grams:{binary_encode_decode.binary_to_ngrams(bin_embedding[2], 3, dictionary)}')


######################## DECODING YURI'S EMBEDDINGS
### embeddings yuri ###############
yuri_embeddings = pd.read_csv('assets/n_gram_embeddings/yuri_n_gram_embedding.tsv', sep='\t')
yuri_embeddings = np.array(yuri_embeddings)
print("YURIIIIIIIIIIIIIIIIIII")
print(type(yuri_embeddings))
print(yuri_embeddings)


#####################
decode = autoencoder.decode()
bin_embedding = decode.predict(yuri_embeddings)
bin_embedding = np.round(bin_embedding, 0)
print(f'Original Embedding: {yuri_embeddings[2351]}')
print(f'Predicted Binary embedding: {bin_embedding[2]}')
original_binary_embedding = n_grams_df['embedding'][2]
print(f'Original Binary embedding: {original_binary_embedding}')
print(f'Original N-grams: {n_grams[2]}')
print(f'Predicted N-grams:{binary_encode_decode.binary_to_ngrams(bin_embedding[2], 3, dictionary)}')