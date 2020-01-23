import re
import json

def get_stanford_core_data(sentences, stanford_nlp):
    """Fills in the stanford core values needed for the model."""
    result = []
    for item in sentences:
        data = dict()
        data['sentence'] = item['sentence']
        data['position'] = item['position'] #from spacy so has 0 index - character offsets
        data['golden-entity-mentions'] = item['golden-entity-mentions'] #from spacy so has 0 index - character offsets
        data['golden-event-mentions'] = []
        try:
            nlp_res_raw = stanford_nlp.annotate(item['sentence'])
            nlp_res = json.loads(nlp_res_raw)
        except Exception as e:
            print('[Warning] StanfordCore Exception: ', e, 'This sentence will be ignored.')
            print('If you want to include all sentences, please refer to this issue: https://github.com/nlpcl-lab/ace2005-preprocessing/issues/1')
            continue
        tokens = nlp_res['sentences'][0]['tokens'] #from stanford - has 1 index
        data['stanford-colcc'] = []
        for dep in nlp_res['sentences'][0]['enhancedPlusPlusDependencies']:
            data['stanford-colcc'].append('{}/dep={}/gov={}'.format(dep['dep'], dep['dependent'] - 1, dep['governor'] - 1))

        data['words'] = list(map(lambda x: x['word'], tokens))
        data['pos-tags'] = list(map(lambda x: x['pos'], tokens))
        data['lemma'] = list(map(lambda x: x['lemma'], tokens))
        data['parse'] = nlp_res['sentences'][0]['parse']
        result.append(data)
    return result

def find_token_index(tokens, start_pos, end_pos, phrase):
    start_idx, end_idx = -1, -1
    for idx, token in enumerate(tokens):
        if token['characterOffsetBegin'] <= start_pos:
            start_idx = idx
#    assert start_idx != -1, "start_idx: {}, start_pos: {}, phrase: {}, tokens: {}".format(start_idx, start_pos, phrase, tokens)
    chars = ''

    def remove_punc(s):
        s = re.sub(r'[^\w]', ' ', s)
        return s

    for i in range(0, len(tokens) - start_idx):
        chars += remove_punc(tokens[start_idx + i]['originalText'])
        if remove_punc(phrase) in chars:
            end_idx = start_idx + i + 1
            break

#    assert end_idx != -1, "end_idx: {}, end_pos: {}, phrase: {}, tokens: {}, chars:{}".format(end_idx, end_pos, phrase, tokens, chars)
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

def verify_result(data):
    def remove_punctuation(s):
        for c in ['-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-', '\xa0']:
            s = s.replace(c, '')
        s = re.sub(r'[^\w]', '', s)
        return s

    def check_diff(words, phrase):
        return remove_punctuation(phrase) not in remove_punctuation(words)

    for item in data:
        words = item['words']
        for entity_mention in item['golden-entity-mentions']:
            if check_diff(''.join(words[entity_mention['start']:entity_mention['end']]), entity_mention['text'].replace(' ', '')):
                print('============================')
                print('[Warning] entity has invalid start/end')
                print('Expected: ', entity_mention['text'])
                print('Actual:', words[entity_mention['start']:entity_mention['end']])
                print('start: {}, end: {}, words: {}'.format(entity_mention['start'], entity_mention['end'], words))

    print('Complete verification')
