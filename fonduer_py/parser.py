from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser import Parser
from fonduer.parser.models import Document, Sentence

from . import config

def parse_dataset(train=False,first_time=False):    
    if train :
        session = config.init_session(config.conn_string_train)
    else:
        session = config.init_session(config.conn_string_predict)

    if not first_time:        
        pass 
    
    elif train:
        doc_preprocessor = HTMLDocPreprocessor(config.train_docs_path)
        corpus_parser = Parser(session, structural=False, lingual=True, tabular = False, replacements=[('\n', ' ')],language='en_core_web_lg')
        corpus_parser.apply(doc_preprocessor, parallelism=config.PARALLEL)
        
        
    else:
        doc_preprocessor = HTMLDocPreprocessor(config.predict_docs_path)
        corpus_parser = Parser(session, structural=False, lingual=True, tabular = False, replacements=[('\n', ' ')],language='en_core_web_lg')
        corpus_parser.apply(doc_preprocessor, parallelism=config.PARALLEL)
    
    return {'document_count': session.query(Document).count(),'sentence_count':session.query(Sentence).count(),'docs':session.query(Document).order_by(Document.name).all(),'session':session}

