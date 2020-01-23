import sys
import os
import logging
import json
import spacy

from stanfordcorenlp import StanfordCoreNLP

from preproccode.preproc.preprocess import preprocess


# spacy
spacy_nlp = spacy.load('en_core_web_lg', disable=["tagger", "parser"])
max_length = os.getenv("MAX_DOCUMENT_LENGTH")
if max_length:
    spacy_nlp.max_length = int(max_length)

sentencizer = spacy_nlp.create_pipe("sentencizer")
spacy_nlp.add_pipe(sentencizer)


# stanford
class StanfordNLP:
    """Getting Stanford running with necessary annotators"""
    def __init__(self, host='http://localhost', port=29000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=60000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,parse',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def annotate(self, sentence):
        return self.nlp.annotate(sentence, properties=self.props)


stanford_nlp = StanfordNLP()

cdr_path = '/home/max/data/wm/document-drops/november/fe844dfe2b5891691be1b65b26c0e184.cdr'

with open(cdr_path, 'r') as in_json:
    cdr_doc = json.load(in_json)
    procced = preprocess(cdr_doc, spacy_nlp, stanford_nlp)
    print(json.dumps(procced))
