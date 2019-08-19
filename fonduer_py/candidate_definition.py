from fonduer.candidates.models import candidate_subclass


def get_candidate_list(mention_list):
    # defining the relation
    person_cand = candidate_subclass('person_cand',[mention_list[0]])
    return [person_cand]