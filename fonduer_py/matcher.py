from fonduer.candidates.matchers import LambdaFunctionMatcher

def person_name_matcher(mention):
    mention_set = set(mention.sentence.ner_tags)
    if len(mention_set) == 1 and 'PERSON' in mention_set:
        return True
    else:
        return False

person_name_function = LambdaFunctionMatcher(func=person_name_matcher)


def get_matchers():
    return [person_name_function]
