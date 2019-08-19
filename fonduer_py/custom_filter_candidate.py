from flair.models import SequenceTagger
from flair.data import Sentence as fl_sen

# Loading the flair tagger
tagger = SequenceTagger.load('ner')

# Function uses NER by Flair to further recognize persons from the extracted candidates and remove other candidates
def is_person(sentence):    
    sentence_token = fl_sen(sentence)
    tagger.predict(sentence_token)
    tagged_dict = sentence_token.to_dict(tag_type='ner')
    for entity in tagged_dict['entities']:
        if entity['type'] == 'PER':
            return True
        else :
            return False
    return False

# Custom filter 
# Usage :
# Removing partial candidates and taking only the longest candidate
# For example: Anil,Anil Kumar, Anil Kumar Singh are the 3 candidates then only Anil Kumar Singh is taken else are removed
#
# Input :
# list of candidates
#
# Returns :
# new list of candidates and ids of those candidates
def filter(candidate_extractor_output, first_time=False):
    session = candidate_extractor_output['session']
    cand_table = candidate_extractor_output['candidate_variable'][0]
    candidate_list = session.query(cand_table).all()

    required_candidate_ids=[]

    if first_time:
        new_train_cands_person=[]
        temp_list = []
        id_dict = dict()
        for mention in candidate_list:
            key = int(mention[0].context.sentence.id) 
            if key not in list(id_dict.keys()):
                id_dict[key] = mention
            else:
                if len((id_dict[key]).get_mentions()[0][0].get_span()) < len(mention.get_mentions()[0][0].get_span()):
                    id_dict[key] = mention
                    
        for value in id_dict.values():
            if is_person(value.get_mentions()[0][0].get_span()):
                temp_list.append(value)
                required_candidate_ids.append(value.id)
            
        new_train_cands_person.append(temp_list)
        return delete_candidates(candidate_extractor_output,cand_table,required_candidate_ids)
    
    else:
        candidate_extractor_output['filtered_candidate_count'] = session.query(cand_table).count()
        return candidate_extractor_output

def delete_candidates(candidate_extractor_output,cand_table,required_candidate_ids):
    session = candidate_extractor_output['session']
    stmt = cand_table.__table__.delete().where(cand_table.id.notin_(required_candidate_ids))
    session.execute(stmt)
    candidate_extractor_output['filtered_candidate_count'] = session.query(cand_table).count()
    return candidate_extractor_output
    
    