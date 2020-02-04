import fasttext


class WordEmbedding:
    """
    FastText word embedding (https://fasttext.cc/)
    """

    def __init__(self, **kwargs):
        """
        Load an already learning word embedding from a bin file

        Args:
            filename: the bin file with the word embedding
            model: an already loaded fasttext model
        """
        if "filename" in kwargs:
            self.model = fasttext.load_model(kwargs["filename"])
        elif "model" in kwargs:
            self.model = kwargs["model"]
        else:
            raise ValueError("At least one between filename and model must not be empty")

    @classmethod
    def learn_embedding(cls, corpus):
        """
        Create a new word embedding from a corpus
        Args:
            corpus: list of words to be inserted in the word embedding

        Returns:
            a new WordEmbedding representing the corpus
        """
        model = fasttext.train_unsupervised(corpus,
                                            model='skipgram',
                                            minCount=1)
        return cls(model=model)

    def save_model(self, saveto):
        self.model.save_model(saveto)

    def get_nn_word(self, word, k=10):
        nn = self.model.get_nearest_neighbors(word, k)
        return [n[1] for n in nn]
