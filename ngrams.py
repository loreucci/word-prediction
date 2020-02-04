from preprocessing import process_text
import math


class NGrams:
    """
    A collection of n-grams gathered from a text corpus
    """

    def __init__(self, corpus_file, n, embedding=None):
        """
        Creates the n-grams for a corpus of sentence and stores them for retrival

        Args:
            corpus_file: file with sentences
            n: size of the n-grams that should be computed (1 for unigrams)
            embedding: a WordEmbedding model to search for matches (optional)
        """
        self.corpus_file = corpus_file
        self.n = n
        self.embedding = embedding

        # unigrams
        if n == 1:

            self.word_freq = {}
            self.word_count = 0
            with open(corpus_file) as corpus:
                for sent in corpus:
                    # remove newline
                    if sent[-1] == '\n':
                        ssent = sent[:-1]
                    else:
                        ssent = sent
                    ssent = ssent.split(' ')
                    for w in ssent:
                        if w not in self.word_freq:
                            self.word_freq[w] = 1
                            self.word_count += 1
                        else:
                            self.word_freq[w] += 1

        else:

            # this will temporarly store the occurences of every N-gram
            ngrams_count = dict()

            # {(N-1)-gram : (word, occurences)} dictionary
            self.ngrams_rev = dict()
            with open(corpus_file) as corpus:
                for sent in corpus:
                    # remove newline
                    if sent[-1] == '\n':
                        ssent = sent[:-1]
                    else:
                        ssent = sent
                    ssent = ssent.split(' ')
                    for i in range(0, len(ssent) - n + 1):
                        ngram = tuple(ssent[i:i + n])  # N-gram
                        if ngram not in ngrams_count:
                            ngrams_count[ngram] = 1
                        else:
                            ngrams_count[ngram] += 1
                        lgram = ngram[:-1]  # (N-1)-gram
                        if lgram not in self.ngrams_rev:
                            self.ngrams_rev[lgram] = []

            # populate dictionary from the temporary count
            for ng, ngc in ngrams_count.items():
                self.ngrams_rev[ng[:-1]].append((ng[-1], ngc))

        # shorter N-grams, to compute perplexity recursively (computed on-demand)
        self.lgrams = None

    def get_predictions(self, lgram):
        """
        Get a list of prediction from a (N-1)-gram.
        If there is no exact match and a WordEmbedding is used, the prediction will be searched also N-grams
        with similar words.

        Args:
            lgram: the (N-1)-gram providing context

        Returns:
            a list of predicted words
        """
        if self.n == 1:
            return []

        if len(lgram) != self.n-1:
            return []
        try:
            preds = self.ngrams_rev[lgram]
        except KeyError:
            if self.embedding is not None:
                similar = []
                for w in lgram:
                    similar.append([w] + self.embedding.get_nn_word(w))
                M = len(similar[0])  # all lists have the same length
                for i in range(int(pow(M, self.n - 1))):
                    idx = [i // int(pow(M, n)) % M for n in range(self.n - 1)]
                    new_lgram = [similar[j][idx[j]] for j in range(len(idx))]
                    try:
                        preds = self.ngrams_rev[tuple(new_lgram)]
                    except KeyError:
                        continue
                    preds.sort(key=lambda x: x[1], reverse=True)
                    return [p[0] for p in preds]
                return []
            else:
                return []
        preds.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in preds]

    def update(self, sent):
        """
        Update the N-grams with text from a new sentences

        Args:
            sent: the new sentence
        """
        if self.n == 1:

            for w in sent:
                if w not in self.word_freq:
                    self.word_freq[w] = 1
                    self.word_count += 1
                else:
                    self.word_freq[w] += 1

        else:

            sent = process_text(sent)
            if sent[-1] == '\n':
                ssent = sent[:-1]
            else:
                ssent = sent
            ssent = ssent.split(' ')
            for i in range(0, len(ssent) - self.n + 1):
                ngram = tuple(ssent[i:i + self.n])
                lgram = ngram[:-1]
                if lgram not in self.ngrams_rev:
                    self.ngrams_rev[lgram] = [(ngram[-1], 1)]
                else:
                    for j, ng in enumerate(self.ngrams_rev[lgram]):
                        if ng[0] == ngram[-1]:
                            self.ngrams_rev[lgram][j] = (ng[0], ng[1]+1)
                            break
                    else:
                        self.ngrams_rev[lgram].append((ngram[-1], 1))

    def perplexity(self, ngram):
        """
        Compute the perplexity of a N-gram

        Args:
            ngram: list of words

        Returns:
            (perplexity of the N-gram, probability of the N-gram, N-gram count)
        """

        if len(ngram) != self.n:
            raise ValueError("Length mismatch when evaluating perplexity")

        # unigrams
        if self.n == 1:
            if ngram[0] in self.word_freq:
                p = self.word_freq[ngram[0]] / self.word_count
                return 1 / p, p, self.word_freq[ngram[0]]
            return math.inf, 0, 0

        # check if ngram is stored
        lgram = ngram[:-1]
        if lgram in self.ngrams_rev:

            for w in self.ngrams_rev[lgram]:

                if w[0] == ngram[-1]:

                    # create (N-1)-grams on demand
                    if self.lgrams is None:
                        self.lgrams = NGrams(self.corpus_file, self.n - 1)

                    # recursively compute
                    _, lprob, lcount = self.lgrams.perplexity(lgram)
                    p = lprob * w[1] / lcount

                    return math.pow(1 / p, 1 / self.n), p, w[1]
            else:
                return math.inf, 0, 0
        else:
            return math.inf, 0, 0

    def average_perplexity(self):
        """
        Computes the average perplexity of the stored N-grams

        Returns:
            average perplexity
        """
        perpl = 0
        ngram_count = 0

        if self.n == 1:
            for w in self.word_freq.keys():
                perpl += self.perplexity((w,))[0]
            ngram_count = self.word_count
        else:
            for lgram, ws in self.ngrams_rev.items():
                for w in ws:
                    perpl += self.perplexity(lgram + (w[0],))[0]
                    ngram_count += 1
        return perpl / ngram_count
