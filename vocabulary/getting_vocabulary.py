import math 

class GettingVocabulary():

    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = None
    
    def get_vocab(self):
        self.vocab = list(sorted(set(' '.join(self.corpus).split())))
        return self.vocab
    
    def get_vocab_to_indx(self):
        self.vocab_to_indx = {word: indx for indx, word in enumerate(self.vocab)}
        return self.vocab_to_indx
    
    def get_indx_to_vocab(self):
        self.indx_to_vocab = {indx: word for word, indx in self.vocab_to_indx.items()}
        return self.indx_to_vocab
    
    def get_binary_rep(self, vocab_to_indx):
        self.vocab_to_indx = vocab_to_indx
        vocab_size = len(self.vocab_to_indx)
        bits = math.ceil(math.log2(vocab_size))

        vocab_to_bin = {word: format(idx, f'0{bits}b') for word, idx in self.vocab_to_indx.items()}
        # bin_to_word = {v: k for k, v in vocab_to_bin.items()}
    
        return vocab_to_bin #, bin_to_word, bits