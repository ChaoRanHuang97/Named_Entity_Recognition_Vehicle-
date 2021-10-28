import spacy
from flask import Flask, request
from model import ner_nlp
from datetime import datetime
from similarity import similarities
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from phone_number_format import format_phone_number

now = str(datetime.utcnow())
app = Flask(__name__)
nlp = ner_nlp(model_path='models/mca-ner-draft_Adam_nlp.model')
nlp_similarity = spacy.load('en_core_web_md')

inf = list(nlp_similarity.Defaults.infixes)
inf = [x for x in inf if '-|–|—|--|---|——|~' not in x] # remove the hyphen-between-letters pattern from infix patterns
infix_re = compile_infix_regex(tuple(inf))

def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

nlp_similarity.tokenizer = custom_tokenizer(nlp_similarity)

search = similarities(nlp_similarity)

@app.route('/')
def index():
    return 'Web App with Python Flask!'

@app.route('/NER', methods=['GET', 'POST'])
def ner():
    posting_request = request.get_json()
    posting = posting_request['posting']
    prediction = nlp.detect(posting)
    MAKE = prediction.get('MAKE', '')
    if MAKE.lower() == 'chevy':
        MAKE = 'CHEVROLET'
    if MAKE.lower() == 'vw':
        MAKE = 'VOLKSWAGEN'
    MODEL = prediction.get('MODEL', '')
    TRIM = prediction.get('TRIM', '')
    VIN = prediction.get('VIN')
    PHONE = prediction.get('PHONE')
    main_doc = MAKE + ' | ' + MODEL + ' | ' + TRIM
    result = search.search(main_doc, prediction.get('YEAR', 2020))
    if VIN != None:
        result['VIN'] = VIN
    if PHONE != None:
        PHONE = format_phone_number(PHONE)
        if PHONE != None:
            result['PHONE'] = PHONE
    print('==========result============')
    print(prediction, result)
    print('============================')
    return {'status_code':200, 'prediction.entities': prediction, 'result': result}

if __name__ == '__main__':
    app.run(host='10.128.0.2', port=8000, debug=True)