#! /usr/bin/env python
# encoding: utf-8

from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion

from ngrams import NGrams

from preprocessing import process_text
from word_embedding import WordEmbedding

import argparse


# parsing arguments
parser = argparse.ArgumentParser("Magic keyboard")
parser.add_argument('-n', type=int, default=3)
parser.add_argument('--embedding', action="store_true")
args = parser.parse_args()

# load N-grams with optional embedding
we_model = None
if args.embedding:
    we_model = WordEmbedding(filename="data/wikipediaML_embedding.bin")
ngrams = NGrams("data/wikipediaML_sentences", args.n, we_model)


def predict_next_words(context_words, current_word):
    """
    Predict the most likely next words the user could type in order of probability.

    Arguments:

    context_words -- array of strings corresponding to the words in the current sentence, not including the word at the user caret
    current_word -- current word at the user caret

    Returns:

    predicted_words -- array of strings corresponding to predictions
    """
    return list(filter(lambda x: x.startswith(current_word or ""),
                       ngrams.get_predictions(tuple(context_words[-(args.n-1):]))))[0:3]


class WordCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        text = process_text(text) + (" " if text[-1] == " " else "")
        context_words = list(filter(lambda s: len(s) > 0, text.split(' ')))
        if len(text) == 0 or text[-1] == ' ' or len(context_words) == 0:
            current_word = ""
        else:
            current_word = context_words[-1]
        if len(current_word) > 0:
            context_words = context_words[:-1]
        for word in predict_next_words(context_words, current_word):
            yield Completion(word, start_position=-len(current_word))


if __name__ == '__main__':
    print('Welcome to magic keyboard')
    while True:
        answer = prompt("> ", completer=WordCompleter())
        if answer == 'q':
            break
        ngrams.update(answer)
