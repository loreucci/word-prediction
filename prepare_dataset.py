from preprocessing import get_sentences_from_text

import xml.etree.ElementTree as ET
import argparse
import nltk


if __name__ == "__main__":

    # download punkt
    nltk.download("punkt")

    parser = argparse.ArgumentParser("predictive keyboard")
    parser.add_argument('--embedding', action="store_true", help="Create word embedding")
    args = parser.parse_args()

    # parse xml to get content of pages
    tree = ET.parse('data/Wikipedia-20200202233538.xml')
    root = tree.getroot()
    corpus = []
    for page in root.findall('{http://www.mediawiki.org/xml/export-0.10/}page'):
        revision = page.find('{http://www.mediawiki.org/xml/export-0.10/}revision')
        text = revision.find('{http://www.mediawiki.org/xml/export-0.10/}text')
        page_content = text.text
        # process pages to get sentences
        corpus += get_sentences_from_text(page_content)

    # create dataset for word embedding (only words)
    with open("data/wikipediaML_words", 'w') as f:
        for s in corpus:
            for w in s.split(' '):
                f.write(w + " ")

    # create dataset for N-grams (with sentences)
    with open("data/wikipediaML_sentences", 'w') as f:
        for s in corpus:
            f.write(s)
            f.write('\n')

    if args.embedding:
        from word_embedding import WordEmbedding

        # learn word embedding
        we = WordEmbedding.learn_embedding("data/wikipediaML_words")
        we.save_model("data/wikipediaML_embedding.bin")
