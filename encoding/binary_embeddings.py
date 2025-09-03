import pandas as pd

class LambdaGramEmbeddings:
    def __init__(self, word_to_indx):
        self.word_to_indx = word_to_indx

    def _embed_lambda_gram(self, words):
        binaries = [self.word_to_indx[word] for word in words if word in self.word_to_indx]
        return "".join(binaries)

    def get_embeddings_df(self, lambda_grams):
        self.lambda_grams = lambda_grams
        data = []
        for lg in self.lambda_grams:
            # separar palabras
            words = lg[0].split() if isinstance(lg[0], str) else lg  
            embedding = self._embed_lambda_gram(words)
            data.append((" ".join(words), embedding))
        return pd.DataFrame(data, columns=["lambda_gram", "embedding"])
