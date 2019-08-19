import torch
from fonduer.learning import LogisticRegression, SparseLogisticRegression, LSTM
import numpy as np

from . import config


ABSTAIN = 0
FALSE = 1
TRUE = 2


def train_model(algorithm_chosen,labeller_output,first_time=False):
    try:
        if first_time:
            # Loading all the elements
            session = labeller_output['session']
            cands = labeller_output['candidate_variable']
            featurizer = labeller_output['featurizer_variable']
            train_marginals = labeller_output['train_marginals']            

            if algorithm_chosen == 'logistic_regression':
                disc_model = LogisticRegression()
            elif algorithm_chosen == 'sparse_logistic_regression':
                disc_model = SparseLogisticRegression()
            else:        
                disc_model = LSTM()

            cand_list = [session.query(cands[0]).all()]
            cand_feature_matrix = featurizer.get_feature_matrices(cand_list)
            disc_model.train((cand_list[0], cand_feature_matrix[0]), train_marginals, n_epochs=1000, lr=0.001)
            disc_model.save(model_file=algorithm_chosen, save_dir=config.base_dir+'/checkpoints', verbose=True)

        return("Trained succesfully",200,)
    except Exception as e:
        print(e)
        return ("Something went wrong",500)

def load_model_and_predict(algorithm_chosen,featurizer_output):    
    session = featurizer_output['session']
    cands = featurizer_output['candidate_variable']
    featurizer = featurizer_output['featurizer_variable']
    
    if algorithm_chosen == 'logistic_regression':
        disc_model = LogisticRegression()
    elif algorithm_chosen == 'sparse_logistic_regression':
        disc_model = SparseLogisticRegression()
    else:        
        disc_model = LSTM()
    
    # Manually load settings and cardinality from a saved trained model.
    checkpoint = torch.load(config.base_dir+'/checkpoints/'+algorithm_chosen)
    disc_model.settings = checkpoint["config"]
    disc_model.cardinality = checkpoint["cardinality"]

    # Build a model using the loaded settings and cardinality.
    disc_model._build_model()
    
    disc_model.load(model_file=algorithm_chosen, save_dir=config.base_dir+'/checkpoints')
    


    cand_list = [session.query(cands[0]).all()]
    cand_feature_matrix = featurizer.get_feature_matrices(cand_list)
    
    test_score = disc_model.predict((cand_list[0], cand_feature_matrix[0]), b=0.5,pos_label=TRUE)
    true_pred = [cand_list[0][_] for _ in np.nditer(np.where(test_score==TRUE))]
    return true_pred
