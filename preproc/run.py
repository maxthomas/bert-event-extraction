import sys
import os
import logging
import json
import spacy
from flask import Flask,render_template,url_for,request, jsonify

from stanfordcorenlp import StanfordCoreNLP
from tokens import find_token_index, fix_entity_index, fix_entity_indices, get_stanford_core_data

app = Flask(__name__)

STATUS_OK = 'healthy'


nlp = spacy.load('en_core_web_lg', disable=["tagger", "parser"])
max_length = os.getenv("MAX_DOCUMENT_LENGTH")
if max_length:
    nlp.max_length = int(max_length)

sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

def sentence_dict_list(doc):
    """Returns a list of dictionaries for each sentence in a CDR.
    This is just a few of those necessary for the model.
    """
    sentences = []
    for sent in doc.sents:
        sentence_dict = {}
        sentence_dict['sentence'] = sent.text
        sentence_dict['position'] = [sent.start_char, sent.end_char]
        entities = []
        for ent in sent.ents:
            entity_dict = {}
            entity_dict['text'] = ent.text
            entity_dict['position'] = [ent.start_char, ent.end_char]
            entity_dict['entity-type'] = ent.label_
            entities.append(entity_dict)
        sentence_dict['golden-entity-mentions'] = entities
        sentence_dict['golden-event-mentions'] = []
        sentences.append(sentence_dict)
    return sentences


class StanfordNLP:
    """Getting Stanford running with necessary annotators"""
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=60000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,parse',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def annotate(self, sentence):
        return self.nlp.annotate(sentence, properties=self.props)


Snlp = StanfordNLP()


def preprocess(cdr):
    """Given a CDR-like dictionary, preprocess to fit the
    format required by the ACE-based event model.
    Errors if the key extracted_text is not in the input dictionary."""

    txt = cdr['extracted_text']
    logging.debug('incoming text: {}'.format(txt))
    doc = nlp(txt)
    sentences = sentence_dict_list(doc)
    post11 = get_stanford_core_data(sentences, Snlp)
    check_it = fix_entity_indices(post11, Snlp)
    return check_it

@app.route('/api/v1/health', methods=['GET'])
def health():
    out = {}
    out['status'] = STATUS_OK
    return jsonify(out)


@app.route('/api/v1/annotate/cdr',methods=['POST'])
def predict():
    """Like /predict, but only returns the annotation, not the entire document."""
    request_json = request.get_json(force=True)
    ready = preprocess(request_json)
    return jsonify(ready)

if __name__ == '__main__':
    app.run(debug=False,
            host='0.0.0.0',
            use_reloader=False,
            threaded=True,
            )
