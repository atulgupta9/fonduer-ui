from fonduer.candidates import CandidateExtractor

from . import config,candidate_definition

""" 
A mention,m, represents a noun, i.e., a real-world person,place, or thing, which can be grouped and identified by its 
mention type, T . For example, “part number” is a mention type, while “BC546” is a corresponding mention.

A relationship of n mentions is an n-ary relation, R(m1,m2, . . . ,mn),which corresponds to a schema, SR(T1,T2, . . . ,Tn). 

A candidate is an n-ary tuple, c = (m1,m2, . . . ,mn), which represents a potentially correct instance of a relation R. 

For instance, a “part number” and a “price” represent a relation with a schema, SR(T1,T2), where “BC546” and “$1.00” 
represent a candidate, c = (m1,m2), of a 2-ary relation, R(m1,m2).
 """
def get_candidates(mention_extractor_output,first_time=False):
    # Dependencies obtained from mention extractor output
    session = mention_extractor_output['session']
    docs = mention_extractor_output['docs']    
   
    
   
    # Defining the candidates for candidate extraction
    cands = candidate_definition.get_candidate_list(mention_extractor_output['mention_variable'])  
    throttlers = []
    
    # Running the candidate extractor on the parsed docs
    candidate_extractor = CandidateExtractor(session, cands,throttlers=None)
    
    if not first_time:
        # Adding candidate extractor required outputs to the mention_extractor _output_dict
        mention_extractor_output['candidate_count'] = len(candidate_extractor.get_candidates()[0])
        mention_extractor_output['candidate_variable'] = cands
        return mention_extractor_output

    candidate_extractor.apply(docs, parallelism=config.PARALLEL)
    
    # Adding candidate extractor required outputs to the mention_extractor_output dict
    mention_extractor_output['candidate_count'] = len(candidate_extractor.get_candidates()[0])
    mention_extractor_output['candidate_variable'] = cands
    
    return mention_extractor_output