import os


try:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, '../..'))
except NameError:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))
    
    
class DataPaths:
    def __init__(self, base_path='/net/nfs.corp/s2-research/recommender/scidocs_v1'):
        self.base_path = base_path
        
        self.cite_val = os.path.join(base_path, 'cite', 'val.qrel')
        self.cite_test = os.path.join(base_path, 'cite', 'test.qrel')
        
        self.cocite_val = os.path.join(base_path, 'cocite', 'val.qrel')
        self.cocite_test = os.path.join(base_path, 'cocite', 'test.qrel')

        self.coread_val = os.path.join(base_path, 'coread', 'val.qrel')
        self.coread_test = os.path.join(base_path, 'coread', 'test.qrel')
        
        self.coview_val = os.path.join(base_path, 'coview', 'val.qrel')
        self.coview_test = os.path.join(base_path, 'coview', 'test.qrel')
        
        self.mag_train = os.path.join(base_path, 'mag', 'train.csv')
        self.mag_val = os.path.join(base_path, 'mag', 'val.csv')
        self.mag_test = os.path.join(base_path, 'mag', 'test.csv')
        
        self.mesh_train = os.path.join(base_path, 'mesh', 'train.csv')
        self.mesh_val = os.path.join(base_path, 'mesh', 'val.csv')
        self.mesh_test = os.path.join(base_path, 'mesh', 'test.csv')
        
        self.recomm_train = os.path.join(base_path, 'recomm', 'train.csv')
        self.recomm_val = os.path.join(base_path, 'recomm', 'val.csv')
        self.recomm_test = os.path.join(base_path, 'recomm', 'test.csv')
        self.recomm_propensity_scores = os.path.join(base_path, 'recomm', 'propensity_scores.json')
        
        self.paper_metadata = os.path.join(base_path, 'paper_metadata.json')
        
##########################################################################################
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from lightning.classification import LinearSVC


np.random.seed(1)


def classify(X_train, y_train, X_test, y_test):
    """
    Simple classification methods using sklearn framework.
    Selection of C happens inside of X_train, y_train via
    cross-validation. 

    Returns: 
        F1 on X_test, y_test (out of 100)
    """
    estimator = LinearSVC(loss="squared_hinge", random_state=42)
    Cs = np.logspace(-4, 2, 7)
    svm = GridSearchCV(estimator=estimator, cv=3, param_grid={'C': Cs})
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    return 100 * f1_score(y_test, preds, average='macro')



def get_X_y_for_classification(embeddings, train_path, val_path, test_path):
    """
    Given the directory with train/test/val files for mesh classification
        and embeddings, return data as X, y pair
        
    Arguments:
        embeddings: embedding dict
        mesh_dir: directory where the mesh ids/labels are stored
        dim: dimensionality of embeddings

    Returns:
        X, y
    """
    dim = len(next(iter(embeddings.values())))
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    X = defaultdict(list)
    y = defaultdict(list)
    for dataset_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
        for s2id, class_label in dataset.values:
            if s2id not in embeddings:
                X[dataset_name].append(np.zeros(dim))
            else:
                X[dataset_name].append(embeddings[s2id])
            y[dataset_name].append(class_label)
        X[dataset_name] = np.array(X[dataset_name])
        y[dataset_name] = np.array(y[dataset_name])
    return X, y



def load_embeddings_from_jsonl(embeddings_path):
    """Load embeddings from a jsonl file.
    The file must have one embedding per line in JSON format.
    It must have two keys per line: `paper_id` and `embedding`

    Arguments:
        embeddings_path {string} -- path to the embeddings file

    Returns:
        embeddings {dictionary} -- a dictionary where each key is the paper id
                                   and the value is a numpy array 
    """
    embeddings = {}
    with open(embeddings_path, 'r') as f:
        for line in f:
            line_json = json.loads(line)
            embeddings[line_json['paper_id']] = np.array(line_json['embedding'])
    return embeddings



# embeddings for all 
embeddings_path = '/net/nfs.corp/s2-research/recommender/scidocs_v1/paper_metadata_embedded.jsonl'
embeddings = load_embeddings_from_jsonl(embeddings_path)

# all of the data paths
data_paths = DataPaths()

# MAG
mag_embeddings_path = '/net/nfs.corp/s2-research/recommender/embeddings/cite1hop-no-venue-mag.jsonl'
mag_embeddings = load_embeddings_from_jsonl(mag_embeddings_path)
X, y = get_X_y_for_classification(mag_embeddings, data_paths.mag_train, data_paths.mag_val, data_paths.mag_test)
# if the algorithm that procued `embeddings` has nothing to tune we use:
mag_test_f1 = classify(X['train'], y['train'], X['test'], y['test'])
# if it DOES have something to tune, then you can tune it without leakage:
mag_val_f1 = classify(X['train'], y['train'], X['val'], y['val'])

# MeSH
mesh_embeddings_path = '/net/nfs.corp/s2-research/recommender/embeddings/cite1hop-no-venue-mesh.jsonl'
mesh_embeddings = load_embeddings_from_jsonl(mesh_embeddings_path)
X, y = get_X_y_for_classification(mesh_embeddings, data_paths.mesh_train, data_paths.mesh_val, data_paths.mesh_test)
# if the algorithm that procued `embeddings` has nothing to tune we use
mesh_test_f1 = classify(X['train'], y['train'], X['test'], y['test'])

#############################################################################
import operator
import pathlib
import pytrec_eval


def get_metrics(qrel_file, run_file, metrics=('ndcg', 'map')):
    with open(qrel_file, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(run_file, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
        
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, set(metrics))
    results = evaluator.evaluate(run)

    out = {}
    for measure in sorted(metrics):
        res = pytrec_eval.compute_aggregated_measure(
                measure, 
                [query_measures[measure]  for query_measures in results.values()]
            )
        out[measure] = np.round(100 * res, 2)
    return out


def make_run_from_embeddings(qrel_file, embeddings, outfile, topk=5, generate_random_embeddings=False):
    with open(qrel_file) as f_in:
        qrels = [line.strip() for line in f_in]

    # a dict where keys are paper-ids and values are all relevant and
    # non-relevant paper-ids in the qrel
    papers = defaultdict(list)

    # each row is in the following format
    # query-id 0 paper-id [relevance]
    # where relevance is 0=negative or 1=positive
    for line in qrels:
        row = line.split(' ')
        papers[row[0]].append(row[2])

    results = []

    missing_queries = 0
    key_error = 0
    success_candidates = 0
    for pid in papers:
        try:
            if generate_random_embeddings:
                emb_query = np.random.normal(0, 0.67, 200)
            else:
                emb_query = embeddings[pid]
        except KeyError:
            missing_queries += 1
            continue
        if len(emb_query) == 0:
            missing_queries += 1
            continue
        # all embeddings for candidate paper ids in the qrel file
        emb_candidates = []
        candidate_ids = []
        for idx, paper_id in enumerate(papers[pid]):
            try:
                if generate_random_embeddings:
                    emb_candidates.append(np.random.normal(0, 0.67, 200))
                else:
                    emb_candidates.append(embeddings[paper_id])
                candidate_ids.append(paper_id)
                success_candidates += 1
            except KeyError:
                key_error += 1
        # calculate similarity based on l2 norm
        emb_query = np.array(emb_query)

        # trec_eval assumes higher scores are more relevant
        # here the closer distance means higher relevance; therefore, we multiply distances by -1
        distances = [-np.linalg.norm(emb_query - np.array(e))
                     if len(e) > 0 else float("-inf")
                     for e in emb_candidates]

        distance_with_ids = list(zip(candidate_ids, distances))

        sorted_dists = sorted(distance_with_ids, key=operator.itemgetter(1))

        added = set()
        for i in range(len(sorted_dists)):
            # output is in this format: [qid iter paperid rank similarity run_id]
            if sorted_dists[i][0] in added:
                continue
            if i < topk:
                results.append([pid, '0', sorted_dists[i][0], '1', str(np.round(sorted_dists[i][1], 5)), 'n/a'])
            else:
                results.append([pid, '0', sorted_dists[i][0], '0', str(np.round(sorted_dists[i][1], 5)), 'n/a'])
            added.add(sorted_dists[i][0])

    pathlib.Path(outfile).parent.mkdir(parents=True, exist_ok=True)

    with open(outfile, 'w') as f_out:
        for res in results:
            f_out.write(f"{' '.join(res)}\n")


# write out run for a specific qrel
# again, the validation file is only here in case you are tuning a baseline

embeddings_path = '/net/nfs.corp/s2-research/recommender/embeddings/cite1hop-no-venue-qrel-all.jsonl'
embeddings_qrel = load_embeddings_from_jsonl(embeddings_path)
run_path = '/tmp/temp.run'

make_run_from_embeddings(data_paths.coview_test, embeddings_qrel, run_path, topk=5, generate_random_embeddings=False)
coview_results = get_metrics(data_paths.coview_test, run_path, metrics=('ndcg', 'map'))

make_run_from_embeddings(data_paths.coread_test, embeddings_qrel, run_path, topk=5, generate_random_embeddings=False)
coread_results = get_metrics(data_paths.coread_test, run_path, metrics=('ndcg', 'map'))

make_run_from_embeddings(data_paths.cite_test, embeddings_qrel, run_path, topk=5, generate_random_embeddings=False)
cite_results = get_metrics(data_paths.cite_test, run_path, metrics=('ndcg', 'map'))

make_run_from_embeddings(data_paths.cocite_test, embeddings_qrel, run_path, topk=5, generate_random_embeddings=False)
cocite_results = get_metrics(data_paths.cocite_test, run_path, metrics=('ndcg', 'map'))

print(coview_results, coread_results, cite_results, cocite_results)


