import re
import json

def find_token_index(tokens, start_pos, end_pos, phrase):
    start_idx, end_idx = -1, -1
    for idx, token in enumerate(tokens):
        if token['characterOffsetBegin'] <= start_pos:
            start_idx = idx

    assert start_idx != -1, "start_idx: {}, start_pos: {}, phrase: {}, tokens: {}".format(start_idx, start_pos, phrase, tokens)
    chars = ''

    def remove_punc(s):
        s = re.sub(r'[^\w]', '', s)
        return s

    for i in range(0, len(tokens) - start_idx):
        chars += remove_punc(tokens[start_idx + i]['originalText'])
        if remove_punc(phrase) in chars:
            end_idx = start_idx + i + 1
            break

    assert end_idx != -1, "end_idx: {}, end_pos: {}, phrase: {}, tokens: {}, chars:{}".format(end_idx, end_pos, phrase, tokens, chars)
    return start_idx, end_idx


def fix_entity_index(item, stanford_nlp):

    nlp_res_raw = stanford_nlp.annotate(item['sentence'])
    nlp_res = json.loads(nlp_res_raw)
    tokens = nlp_res['sentences'][0]['tokens']

    sent_start_pos = item['position'][0]

    for entity_mention in item['golden-entity-mentions']:

        position = entity_mention['position']

        start_idx, end_idx = find_token_index(
            tokens=tokens,
            start_pos=position[0] - sent_start_pos,
            end_pos=position[1] - sent_start_pos + 1,
            phrase=entity_mention['text'],
    )

        entity_mention['start'] = start_idx
        entity_mention['end'] = end_idx

       # del entity_mention['position']

    #item['golden-entity-mentions'].append(entity_mention)

    return item


def fix_entity_indices(sentences, stanford_nlp):
    
    results = []

    for item in sentences:

        result = fix_entity_index(item, stanford_nlp)

        results.append(result)

    return results
