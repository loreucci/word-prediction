import re
from nltk.tokenize import sent_tokenize


def process_text(text):
    """
    Process some text by removing special characters, unnecessary whitespaces and numbers

    Args:
        text: a string containing text (usually a sentence of the corpus)

    Returns:
        the processed text

    """
    # remove links and replace with text
    re.sub(r'\[\[.*\|(.*)\]\]', r'\1', text)

    # remove math formulas
    text = re.sub(r'<math.*math>', '', text)

    # remove references
    text = re.sub(r'<ref.*ref>', '', text)

    # remove all the special characters
    text = re.sub(r'\W', ' ', str(text))

    # change multiple spaces into single spaces
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # remove numbers
    text = re.sub(r'[0-9]+s?', '', text)

    # remove double spaces and trailing spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s$', '', text)
    text = re.sub(r'^\s', '', text)

    # convert to Lowercase
    text = text.lower()

    return text


def get_sentences_from_text(text):
    """
    Process and separate sentences from a body of text

    Args:
        text: a body of text (str) containing multiple sentences

    Returns:
        list of processed sentences

    """

    # divide into sentences
    sentences = sent_tokenize(text)

    # process sentences one at a time
    proc_sentences = [process_text(sentence) for sentence in sentences if sentence.strip() != '']

    # return only sentences longer than 3 words
    return [s for s in proc_sentences if len(s.split(' ')) >= 3]
