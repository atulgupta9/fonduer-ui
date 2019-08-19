from fonduer.candidates.models import Mention
from fonduer.candidates import MentionExtractor

from . import mention_definition,matcher,mention_space,config


def get_mentions(parser_output,first_time=False):
    # Dependencies obtained from parser output
    session = parser_output['session']
    docs = parser_output['docs']    
   
    if not first_time:
         # Adding mention extractor required outputs to the parser_output_dict
        parser_output['mention_count'] = session.query(Mention).count()
        parser_output['mention_variable'] = mention_definition.get_mention_list()
        return parser_output
   
   
    
    # Defining the mention, mention_space and matchers for mention extraction
    mentions = mention_definition.get_mention_list()
    mention_spaces = mention_space.get_mention_spaces()
    matchers = matcher.get_matchers()    
    
    # Running the mention extractor on the parsed docs
    mention_extractor = MentionExtractor(session,mentions,mention_spaces,matchers)
    mention_extractor.apply(docs, parallelism=config.PARALLEL)
    
    # Adding mention extractor required outputs to the parser_output_dict
    parser_output['mention_count'] = session.query(Mention).count()
    parser_output['mention_variable'] = mentions 
    
    return parser_output