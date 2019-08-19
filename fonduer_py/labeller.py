import pandas as pd
from fonduer.supervision import Labeler
from metal.label_model import LabelModel

from . import config

ABSTAIN = 0
FALSE = 1
TRUE = 2

gold_df = pd.read_csv(config.gold_file_path,sep=';')

def has_ceo(c):
    for index, row in gold_df.iterrows():        
        if c.get_mentions()[0][0].get_span() in str(row['ceo']).strip():
            return TRUE
        else:
            continue
    return FALSE

lfs = [has_ceo]



def apply_labellling_functions(featurizer_output):
    session = featurizer_output['session']
    cands = featurizer_output['candidate_variable']
    labeler = Labeler(session, cands)
    labeler.apply(lfs=[lfs], train=True, parallelism=config.PARALLEL)
    
    train_cands = []
    train_cands.append(session.query(featurizer_output['candidate_variable'][0]).all())
    L_train = labeler.get_label_matrices(train_cands)

    gen_model = LabelModel(k=2)
    gen_model.train_model(L_train[0], n_epochs=300, print_every=100)

    train_marginals = gen_model.predict_proba(L_train[0])
    
    featurizer_output['train_marginals'] = train_marginals
    return featurizer_output
