import logging

from .tokens import find_token_index, fix_entity_index, fix_entity_indices, get_stanford_core_data


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


def preprocess(cdr, spacy_nlp, stanford_nlp):
    """Given a CDR-like dictionary, preprocess to fit the
    format required by the ACE-based event model.
    Errors if the key extracted_text is not in the input dictionary."""

    txt = cdr['extracted_text']
    logging.debug('incoming text: {}'.format(txt))
    txt = txt.replace('\n',' ')
    doc = spacy_nlp(txt)
    sentences = sentence_dict_list(doc)
    post11 = get_stanford_core_data(sentences, stanford_nlp)
    check_it = fix_entity_indices(post11, stanford_nlp)
    return check_it
