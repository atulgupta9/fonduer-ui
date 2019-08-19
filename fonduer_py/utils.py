
import codecs
import csv
from builtins import range

from fonduer.supervision.models import GoldLabel, GoldLabelKey

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm

from . import config

# Define labels
ABSTAIN = 0
FALSE = 1
TRUE = 2


def get_gold_dict(
        filename, doc_on=True, sectionhead_on=True, docs=None
):
    with codecs.open(filename) as csvfile:
        gold_reader = csv.reader(csvfile, delimiter=';')
        #skip header row
        next(gold_reader)
        gold_dict = set()
        #print(docs)
        for row in gold_reader:
            # print(row)
            (doc,company,revenue_tag,revenue,unit,net_income_tag,net_income,ceo,chairman,bods,region,year) = row
#             docname_without_spaces = doc.replace(' ',' ')
            #print(docname_without_spaces)
#             if docs is not None:
#                 docs = [d.replace(' ',' ') for d in docs]
            
#             if docs is None or docname_without_spaces.upper() in docs:
            if docs is None or doc.split('.')[0] in docs:
                if not (doc and ceo):
                    continue
                else:
                    key_ceo = []
                    key_chairman = []
                   
                    key_chairman.append(doc.split('.')[0])
                    key_chairman.append(chairman)
                    # gold_dict.add(tuple(key_chairman))

                    key_ceo.append(doc.split('.')[0])
                    key_ceo.append(ceo)
                    gold_dict.add(tuple(key_ceo))
                    
                    
                    if sectionhead_on:
                        for director in bods.split(','):
                            key=[]
                            key.append(doc.split('.')[0])
                            key.append(director.strip())
                            # gold_dict.add(tuple(key))
    #print("Length",len(gold_dict))
#     print(gold_dict)
    return gold_dict

def load_section_heading_gold_labels(
    featurizer_output,annotator_name="gold"
):
    """
    :param session: The database session to use.
    :param candidate_classes: Which candidate_classes to load labels for.
    :param filename: Path to the CSV file containing gold labels.    
    """
    session = featurizer_output['session']
    candidate_classes = featurizer_output['candidate_variable']
    filename = config.gold_file_path
    
    # Check that candidate_classes is iterable
    candidate_classes = (
        candidate_classes
        if isinstance(candidate_classes, (list, tuple))
        else [candidate_classes]
    )

    ak = session.query(GoldLabelKey).filter(GoldLabelKey.name == annotator_name).first()
    # Add the gold key
    if ak is None:
        ak = GoldLabelKey(
            name=annotator_name,
            candidate_classes=[_.__tablename__ for _ in candidate_classes],
        )
        session.add(ak)
        session.commit()

    # Bulk insert candidate labels
    candidates = []
    for candidate_class in candidate_classes:
        candidates.extend(session.query(candidate_class).all())

    gold_dict = get_gold_dict(filename)
    cand_total = len(candidates)
    print(f"Loading {cand_total} candidate labels")
    labels = 0

    cands = []
    values = []
    for i, c in enumerate(tqdm(candidates)):
        doc = (c[0].context.sentence.document.name)       
        val = (c[0].context.get_span())

        label = session.query(GoldLabel).filter(GoldLabel.candidate == c).first()
        if label is None:
            if (doc, val) in gold_dict:
                values.append(TRUE)
            else:
                values.append(FALSE)

            cands.append(c)
            labels += 1

    # Only insert the labels which were not already present
    session.bulk_insert_mappings(
        GoldLabel,
        [
            {"candidate_id": cand.id, "keys": [annotator_name], "values": [val]}
            for (cand, val) in zip(cands, values)
        ],
    )
    session.commit()

    print(f"GoldLabels created: {labels}")


def entity_confusion_matrix(pred, gold):
    if not isinstance(pred, set):
        pred = set(pred)
    if not isinstance(gold, set):
        gold = set(gold)
    TP = pred.intersection(gold)
    FP = pred.difference(gold)
    FN = gold.difference(pred)
    return (TP, FP, FN)


def entity_level_f1(
    candidates, gold_file, corpus=None
):
    """Checks entity-level recall of candidates compared to gold.
    Turns a CandidateSet into a normal set of entity-level tuples
    (doc, part, [attribute_value])
    then compares this to the entity-level tuples found in the gold.
    Example Usage:
        from hardware_utils import entity_level_total_recall
        candidates = # CandidateSet of all candidates you want to consider
        gold_file = 'tutorials/tables/data/hardware/hardware_gold.csv'
        entity_level_total_recall(candidates, gold_file, 'stg_temp_min')
    """
    docs = [(doc.name) for doc in corpus] if corpus else None    
    
    
    # Turn CandidateSet into set of tuples
    print("Preparing candidates...")
    entities = set()
    for i, c in enumerate(tqdm(candidates)):        
        doc = c[0].context.sentence.document.name.upper()        
        val = c[0].context.get_span()
        entities.add((doc, val))

    gold_set = get_gold_dict(
        gold_file,
        docs=docs
    )

    if len(gold_set) == 0:
        print(f"Gold File: {gold_file}")
        print("Gold set is empty.")
        return    
        
    (TP_set, FP_set, FN_set) = entity_confusion_matrix(entities, gold_set)
    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    prec = TP / (TP + FP) if TP + FP > 0 else float("nan")
    rec = TP / (TP + FN) if TP + FN > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")
    print("========================================")
    print("Scoring on Entity-Level Gold Data")
    print("========================================")
    print(f"Corpus Precision {prec:.3}")
    print(f"Corpus Recall    {rec:.3}")
    print(f"Corpus F1        {f1:.3}")
    print("----------------------------------------")
    print(f"TP: {TP} | FP: {FP} | FN: {FN}")
    print("========================================\n")
    return [sorted(list(x)) for x in [TP_set, FP_set, FN_set]]



def entity_to_candidates(entity, candidate_subset):
    matches = []
    for c in candidate_subset:        
        c_entity = tuple(
            [c[0].context.sentence.document.name.upper()]
            + [c[i].context.get_span() for i in range(len(c))]
        )
        c_entity = tuple([str(x) for x in c_entity])
        if c_entity == entity:
            matches.append(c)
    return matches