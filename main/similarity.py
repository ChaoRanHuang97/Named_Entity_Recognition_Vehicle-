import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import process
from datetime import date
from util import get_predicted_price

CONFIDENCE_STD_LIMIT = 0.02
sell_year, sell_week, _ = date.today().isocalendar()

class similarities():
    def __init__(self, nlp_similarity):
        self.target_dict = pd.read_json('ymmt-combinations-new.json')
        self.target_dict['target'] = self.target_dict['make'] + ' | ' + self.target_dict['model'] + ' | ' + self.target_dict['trim']
        self.makes = set(self.target_dict['make'])
        self.nlp_similarity = nlp_similarity

    def get_similarity(self, search_doc, main_doc):
        search = self.nlp_similarity(search_doc)
        main = self.nlp_similarity(main_doc)
        return main.similarity(search)

    def narrow_search_range(self, match, options):
        str2Match = match
        strOptions = options
        # You can also select the string with the highest matching percentage
        highest = process.extractOne(str2Match, strOptions)
        return highest

    def trim_fuzzy_search(self, match, options):
        str2Match = match
        strOptions = options
        # You can also select the string with the highest matching percentage
        candidates = process.extract(str2Match, strOptions, limit=None)
        if candidates[0][1] > 0:
            print('lds working...')
        return candidates

    def model_process(self, model):
        result = ''
        for idx in range(len(model)):
            if model[idx].isalpha():
                result += model[idx]
            else:
                break
        return result

    def search(self, main_doc, year):
        if main_doc.split(' | ')[1] == '':
            result = {'result': 'TOO FEW INFORMATION'}
            return result
        all_sim = []
        name_list = []
        print('narrowing range...')
        target_make, make_score = self.narrow_search_range(main_doc.split(' | ')[0], self.makes)
        if make_score < 75:
            return {'result': 'TOO FEW INFORMATION'}
        year_narrow = self.target_dict[self.target_dict['year'] == int(year)]
        model_options = set(year_narrow[year_narrow['make'] == target_make]['model'])
        if len(model_options) == 0:
            return {'result': 'TOO FEW INFORMATION'}
        if target_make == 'B M W':
            main_doc = main_doc.split(' | ')
            main_doc.insert(1, main_doc[1][0] + ' SERIES | ' + main_doc[1])
            main_doc = ' | '.join(main_doc).upper()
        if target_make == 'MERCEDES-BENZ':
            main_doc = main_doc.split(' | ')
            model_temp = self.model_process(main_doc[1])
            if len(model_temp) == 1:
                model_temp += ' CLASS | '
                main_doc.insert(1, model_temp  + main_doc[1])
            main_doc = ' | '.join(main_doc).upper()
        target_model, model_score = self.narrow_search_range(main_doc.split(' | ')[1], model_options)
        if make_score <= 60:
            return {'result': 'TOO FEW INFORMATION'}
        make_narrow = self.target_dict[self.target_dict['make'] == target_make]
        model_narrow = make_narrow[make_narrow['model'] == target_model]
        trim_options = list(model_narrow['trim'])
        levenshtein_distance_scores = self.trim_fuzzy_search(main_doc.split(' | ')[2], trim_options)
        for idx, score in enumerate(levenshtein_distance_scores):
            doc = " | ".join([target_make, target_model, score[0]])
            score = (doc, score[1] / 100)
            levenshtein_distance_scores[idx] = score
        target_docs = set(model_narrow ['target'])
        print('getting similarities for : \n', target_make, target_model)
        for target_doc in tqdm(target_docs):
            similarity_score = self.get_similarity(target_doc, main_doc)
            all_sim.append(similarity_score)
            name_list.append(target_doc)
        all_sim = np.array(all_sim)
        idx = (-all_sim).argsort()
        # VALIDATING SEARCH
        print('validating with prices...')
        vector_space_scores = list(zip(np.array(name_list)[idx], all_sim[idx]))
        heuristic_similarities = dict()
        for vss_key, vss_value in vector_space_scores:
            for lds_key, lds_value in levenshtein_distance_scores:
                if lds_key == vss_key and lds_value > 0.3:
                    heuristic_similarities[lds_key] = (lds_value * 0.8+ vss_value * 0.2)
                    break
        if len(heuristic_similarities.items()) == 0:
            heuristic_similarities = dict(vector_space_scores)
        heuristic_similarities = dict(sorted(list(heuristic_similarities.items()), key= lambda x: x[1], reverse=True)[:5])
        final_result = self.validating(heuristic_similarities, year=year)
        return final_result

    def validating(self, similarities, year):
        print(similarities)
        confidence_avg = sum(similarities.values())/len(similarities.values())
        confidence_std = np.std(list(similarities.values()))
        candidates = []
        for key in similarities.keys():
            candidates.append(key.split(' | '))
        if list(similarities.values())[0] >= 0.85:
            result = dict(zip(['MAKE', 'MODEL', 'TRIM'], candidates[0]))
            result['HIGHEST_CONFIDENCE'] = list(similarities.values())[0]
            result['STD'] = confidence_std
            return result
        elif 0.75 <= confidence_avg and confidence_avg < 0.85:
            trim_selection = []
            for idx, candidate in enumerate(candidates):
                make, model, trim = candidate
                price = get_predicted_price(sell_year=sell_year,
                                            sell_week=sell_week,
                                            year=year, make=make,
                                            model=model,
                                            trim=trim)
                confidence = list(similarities.values())[idx]
                trim_selection.append((price, confidence, idx))
            trim_selection.sort(key=lambda x: (x[0], x[1]), reverse=True)
            selection_idx = int( (1 + len(trim_selection)) / 2) - 1
            result = dict(zip(['MAKE', 'MODEL', 'TRIM'], candidates[trim_selection[selection_idx][2]]))
            result['AVERAGE_CONFIDENCE'] = confidence_avg
            result['STD'] = confidence_std
            return result
        elif list(similarities.values())[0] < 0.85 and list(similarities.values())[0] > 0.4:
            result = dict(zip(['MAKE', 'MODEL', 'TRIM'], candidates[0]))
            result['HIGHEST_CONFIDENCE'] = list(similarities.values())[0]
            result['STD'] = confidence_std
            return result
        else:
            result = {'result': 'TOO FEW INFORMATION'}
            result['AVERAGE_CONFIDENCE'] = confidence_avg
            result['STD'] = confidence_std
            return result

if __name__ == '__main__':
    from word2number import w2n
    print(w2n.word_to_num('2 point three'))
