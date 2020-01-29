import re
import string
import itertools
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


class Preprocessing:

    def __init__(self):
        pass

    def cleaning(self, texts, html=True, lower=True, newline=True, punctuation=True, number=True):
        """
        Parameters
        ----------
        texts : list
            text of list
        html : bool
            If true, you can eliminate html tag
        lower : bool
            If true, texts are changed lowercase
        newline : bool
            If true, you can eliminate newline tag
        punctuation : bool
            If true, you can eliminate symbols without "." and "?"
        number : bool
            If true, numbers are changed to "0"
        """
        clean_texts = texts
        if html:
            clean_texts = [BeautifulSoup(text, 'html.parser')
                           for text in clean_texts]
            clean_texts = [text.get_text() for text in clean_texts]
        if lower:
            clean_texts = [text.lower() for text in clean_texts]
        if newline:
            clean_texts = ["".join(text.splitlines()) for text in clean_texts]
        if punctuation:
            alphabet = re.compile(r"[^a-zA-Z0-9.? ]")
            clean_texts = [alphabet.sub("", text) for text in clean_texts]
        if number:
            number = re.compile(r"[0-9]+")
            clean_texts = [number.sub("0", text) for text in clean_texts]
        clean_texts = [re.sub("  ", " ", text) for text in clean_texts]
        return clean_texts

    def get_words(self, text, stop=None):
        stop_words = stopwords.words("english")
        if stop:
            for stopword in stop:
                stop_words.append(stopword)
        text = re.sub(r"[.?]", "", text)
        tokens = word_tokenize(text)
        words = [word for word in tokens if word not in stop_words]
        return words

    def get_sentences(self, text):
        return sent_tokenize(text)

    def get_docs(self, texts):
        """
        Making words set in each documents.

        Return
        ------
        Words in each documents
        """
        docs = [self.get_words(text) for text in texts]
        docs = list(filter(lambda x: x != [], docs))
        return docs

    def get_corpus(self, docs):
        """
        Making the corpus.

        Return
        ------
        Corpus in all documents.
        """
        word2num = dict()
        num2word = dict()
        count = 0
        for d in docs:
            for w in d:
                if w not in word2num.keys():
                    word2num[w] = count
                    num2word[count] = w
                    count += 1
        return word2num, num2word

    def get_ndocs(self, docs, word2num):
        """
        Numerralization the docs
        """
        ndocs = [[word2num[w] for w in d] for d in docs]
        return ndocs

    def get_wordset(self, docs):
        wordset = list(itertools.chain(*docs))
        return wordset
