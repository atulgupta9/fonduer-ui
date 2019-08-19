import os
from fonduer import Meta
from fonduer.parser import Parser

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

gold_file_path = os.path.join(base_dir,'gold_set.csv')

train_docs_path = os.path.join(base_dir,'train_files')
predict_docs_path = os.path.join(base_dir,'predict_files')

PARALLEL = 8 # assuming a quad-core machine
train_db = "fonduer_pred_train"
predict_db = "fonduer_pred_predict"

conn_string_train = 'postgresql://postgres@localhost/' + train_db
conn_string_predict = 'postgresql://postgres@localhost/' + predict_db

def init_session(conn_string):
    return Meta.init(conn_string).Session()
