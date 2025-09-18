import pandas as pd
import numpy as np
from autoencoder.nn import Autoencoder
from visualization.plotting import Visualization 
import utils

n_gram = '5'
problem = 'software_requirements'
df = pd.read_csv(f'data/{problem}/dataset.csv')
# preprocessing the dataset
preprocessed_df = utils.preprocessing(df, 'plus')
#getting the vocabulary, vocab_to_index and vocab_to_binary
word_to_bin, vocab_to_index = utils.get_vocab_ind_bin(preprocessed_df, output_file=f'assets/method/{problem}')

l_grams = utils.get_lambda_grams(preprocessed_df, int(n_gram))

#getting the lambda grams to binary embeddings
utils.lambda_grams_to_binary(word_to_bin, l_grams, f'assets/bin_embeddings/{problem}/{n_gram}_grams/binary_embeddings.csv')

################## Splitting into training and testing data 
utils.split_data(f'assets/bin_embeddings/{problem}/{n_gram}_grams/binary_embeddings.csv', f'data/{problem}/{n_gram}_grams')

#################### Loading data to train autoencoder
#loading training and testing data for 4_grams
X_train = utils.upload_data_to_train(f'data/{problem}/{n_gram}_grams/X_train.csv', 'embedding')
X_test = utils.upload_data_to_train(f'data/{problem}/{n_gram}_grams/X_test.csv', 'embedding')
X_val = utils.upload_data_to_train(f'data/{problem}/{n_gram}_grams/X_val.csv', 'embedding')
y_train, y_test, y_val = X_train, X_test, X_val

dictionary = utils.upload_vocab_to_binary_dictionary(file=f'assets/method/{problem}/vocab_to_binary.csv')
##################### Step 8: Creating and training the Neural Network (Autoencoder)

initialize_weights_file = f'assets/weights/{problem}/{n_gram}_grams/initial_weights.pkl'
autoencoder = Autoencoder(X_train.shape[1], y_train.shape[1], embedding_size=200)
autoencoder.save_initialize_weights(initialize_weights_file=initialize_weights_file)
history = autoencoder.fit(X_train, y_train, X_val, y_val, epochs=200, batch_size=32)
autoencoder.save(f'assets/models/{problem}/{n_gram}_grams/model_{n_gram}.h5')

##################### Step 9: Visualizing the training plots
plot = Visualization()
plot.plotting_metric(history.history, 'cosine_similarity', 'val_cosine_similarity', path=f'assets/learning_graphs/{problem}/{n_gram}_grams', fig_name='Learning training')
plot.plotting_loss(history.history, 'loss', 'val_loss', path=f'assets/learning_graphs/{problem}/{n_gram}_grams', fig_name='Loss training')

#################### Step 10: Predicting
#comment following line if you do not want to predict
autoencoder = Autoencoder(X_train.shape[1], y_train.shape[1])
model = autoencoder.load_model(f'assets/models/{problem}/{n_gram}_grams/model_{n_gram}.h5')
y_pred = model.predict(X_test)
y_pred = y_pred.round(0)


#################### Step 11: Encode

encode = autoencoder.encode()
n_grams_embeddings = encode.predict(X_test)
df_embeddings = pd.DataFrame(n_grams_embeddings)
df_embeddings.to_csv(f'assets/n_gram_embeddings/{problem}/{n_gram}_grams/n_gram_embedding.tsv', sep='\t', index=False, header=False)

n_grams_df = pd.read_csv(f'data/{problem}/{n_gram}_grams/X_test.csv')
n_grams = n_grams_df['lambda_gram']
n_grams.to_csv(f'assets/n_gram_embeddings/{problem}/{n_gram}_grams/n_gram_words.tsv', sep='\t', index=False, header=False)

##################### Step 12: Decode
ind = 2
n_grams_df = pd.read_csv(f'data/{problem}/{n_gram}_grams/X_test.csv')
n_grams = n_grams_df['lambda_gram']
n_grams_embeddings = np.loadtxt(f'assets/n_gram_embeddings/{problem}/{n_gram}_grams/n_gram_embedding.tsv', delimiter='\t')

autoencoder = Autoencoder(X_train.shape[1], y_train.shape[1])
model= autoencoder.load_model(f'assets/models/{problem}/{n_gram}_grams/model_{n_gram}.h5')
decode = autoencoder.decode()

bin_embedding = decode.predict(n_grams_embeddings)
bin_embedding = np.round(bin_embedding, 0)
# print(f'Original Embedding: {n_grams_embeddings[ind]}')
original_binary_embedding = n_grams_df['embedding'][ind]
print(f'Kari: Original Binary embedding: {original_binary_embedding}')
karina_bin_predicted = str(bin_embedding[ind]).replace('[', '').replace('.','').replace(' ','').replace(']','').replace('\n','')
print(f'Kari: Predicted Binary embedding: {karina_bin_predicted}')
print(f'Kari: Original N-grams: {n_grams[ind]}')
print(f'Kari: Predicted N-grams:{utils.binary_to_ngrams(bin_embedding, ind, int(n_gram), dictionary)}')

exit()
######################## DECODING YURI'S EMBEDDINGS
### embeddings yuri ###############
yuri_embeddings = np.loadtxt(f'assets/n_gram_embeddings/{problem}/{n_gram}_grams/yuri_n_gram_embedding.tsv', delimiter='\t')
bin_embedding = decode.predict(yuri_embeddings)
bin_embedding = np.round(bin_embedding, 0)
# print(f'Original Embedding: {yuri_embeddings[ind]}')
original_binary_embedding = n_grams_df['embedding'][ind]
print(f'Yuri: Original Binary embedding: {original_binary_embedding}')
yuri_bin_predicted = str(bin_embedding[ind]).replace('[', '').replace('.','').replace(' ','').replace(']','').replace('\n','')
print(f'Yuri: Predicted Binary embedding: {yuri_bin_predicted}')
print(f'Yuri: Original N-grams: {n_grams[ind]}')
print(f'Yuri: Predicted N-grams:{utils.binary_to_ngrams(bin_embedding, ind, int(n_gram), dictionary)}')