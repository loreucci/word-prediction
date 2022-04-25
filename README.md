# word-prediction

A predictive keyboard based on n-grams extracted from a wikipedia corpus.

![demo](https://raw.githubusercontent.com/loreucci/word-prediction/master/word-prediction.gif)

### Usage

Before executing the predictive keyboard, the corpus must be processed by executing

`python prepare_dataset.py --embedding`

or

`python prepare_dataset.py`

The first version will also include the generation of the word embedding.

Once the process is finished (it will create new files in the `data` folder), the keyboard can be launched with

`python repl.py -n 3 --embedding`

where `-n` selects the order of the n-grams used and `--embedding` toggles the word embedding.
