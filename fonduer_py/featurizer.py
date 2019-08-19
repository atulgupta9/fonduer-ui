from fonduer.features import Featurizer
from fonduer.features.models import Feature, FeatureKey

import pickle

from . import config

def get_features(candidate_filtered_output,train=False,first_time=False):
    session = candidate_filtered_output['session']
    cands = candidate_filtered_output['candidate_variable']
    
    featurizer = Featurizer(session, cands)
    if first_time:
        if train :
            featurizer.apply(train=train,parallelism=config.PARALLEL)
            key_names = [key.name for key in featurizer.get_keys()]
            with open(config.base_dir+'feature_keys.pkl', 'wb') as f:
                pickle.dump(key_names, f)
        
        else:
            feature_count = session.query(Feature).count()
            feature_key_count = session.query(FeatureKey).count()
            if feature_count > 0 or feature_key_count > 0:
                featurizer.clear_all()
            with open(config.base_dir+'feature_keys.pkl', 'rb') as f:
                key_names = pickle.load(f)

            featurizer.drop_keys(key_names)
            featurizer.upsert_keys(key_names)
            featurizer.apply(train=train,parallelism=config.PARALLEL)

    # Adding featurizer output to the candidate extractor output dict
    candidate_filtered_output['featurizer_variable'] = featurizer
    return candidate_filtered_output