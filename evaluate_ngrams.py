from ngrams import NGrams


if __name__ == '__main__':

    MAXN = 5

    for N in range(1, MAXN+1):
        unigrams = NGrams("data/wikipediaML_sentences", N)
        print("{}-grams: {}".format(N, unigrams.average_perplexity()))
