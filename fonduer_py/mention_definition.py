from fonduer.candidates.models import mention_subclass


def get_mention_list():
    persion_mention = mention_subclass('person_mention')
    return [persion_mention]