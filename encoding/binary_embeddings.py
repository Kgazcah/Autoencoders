import pandas as pd

class LambdaGramEmbeddings:
    def __init__(self, word_to_indx):
        self.word_to_indx = word_to_indx

    def _embed_lambda_gram(self, words):
        binaries = [self.word_to_indx[word] for word in words if word in self.word_to_indx]
        return "".join(binaries)
    
    def _embed_lambda_gram_sep(self, words):
        return [self.word_to_indx[word] for word in words if word in self.word_to_indx]

    def binary_to_ngrams(self, binary_embedding, ngram, dictionary):
        if not isinstance(binary_embedding, str):
            binary_embedding = ''.join(map(str, binary_embedding.astype(int)))
        self.binary = binary_embedding
        n_dim = len(self.binary) // ngram
        words = []
        inverted_dict = {v: k for k, v in dictionary.items()}
        for i in range(ngram):
            start = i * n_dim
            end = start + n_dim
            segment = self.binary[start:end]
            #Looking up in the dictionary
            word = inverted_dict.get(segment, f"<UNK:{segment}>")
            words.append(word)
        return words

    def get_embeddings_df(self, lambda_grams, fun=0):
        self.lambda_grams = lambda_grams
        data = []
        for lg in self.lambda_grams:
            # separar palabras
            words = lg[0].split() if isinstance(lg[0], str) else lg
            if fun == 0:  
                embedding = self._embed_lambda_gram(words)
            elif fun == 1:
                embedding = self._embed_lambda_gram_sep(words)
            data.append((" ".join(words), embedding))
        return pd.DataFrame(data, columns=["lambda_gram", "embedding"])
    

    def get_embeddings_df_to_classify(self, lambda_grams):
        sentences_binary = []
        for lg in lambda_grams:
            embedding = self._embed_lambda_gram(lg)
            sentences_binary.append(embedding)
        return sentences_binary
    
    
      
