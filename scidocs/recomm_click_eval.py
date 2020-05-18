import argparse
import jsonlines
import math
import torch
import os
import subprocess
import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models import archival
from scidocs.paths import DataPaths
import scidocs.recommender.simclick_data_reader
import scidocs.recommender.simpaper_recommender
from scidocs.recommender.simpaper_recommender import SimpaperRecommender
import csv

import json
import shutil

import tqdm

def evaluate_ranking_performance(archive_path, test_data_path, cuda_device):

    archive = archival.load_archive(archive_path, cuda_device=cuda_device)
    params = archive.config
    sr = archive.model

    propensity_score_path = params['model'].params['propensity_score_path']

    with open(propensity_score_path) as f_in:
        adjClickDistribution = torch.Tensor(json.load(f_in)['scores'])

    #adjust to sum to one:
    adjClickDistribution = adjClickDistribution * len(adjClickDistribution) / (
        sum(adjClickDistribution))

    dr = DatasetReader.from_params(params['dataset_reader'])

    sr.eval()

    with open(test_data_path, 'r') as f_in:
        clicks = f_in.read().splitlines()
    clicks = clicks[1:]
    print(f"Testing on {len(clicks)} examples.")

    correct_order = 0
    incorrect_order = 0
    adj_correct_order = 0
    adj_incorrect_order = 0
    ndcg_numerator = 0
    adj_ndcg_numerator = 0
    adj_demonimator = 0
    mrr = 0
    adj_mrr = 0
    rprec = 0
    adj_rprec = 0

    output = []

    for line in tqdm.tqdm(clicks):
        [query_id, clicked_id, other_id_str] = list(csv.reader([line], delimiter=',', quotechar='"'))[0]

        other_ids = other_id_str.strip().split(",")
        position = 1
        paper_score = [] #used for output
        clicked_position = other_ids.index(clicked_id) + 1
        pos_instance = dr.text_to_instance(query_id, clicked_id, clicked_position)
        pos_score = sr.forward_on_instance(pos_instance)['pos_score']
        paper_score.append([clicked_id, pos_score])
        neg_scores = []
        adj = 1 / adjClickDistribution[clicked_position-1]
        for other_id in other_ids:
            if not other_id == clicked_id:
                neg_instance = dr.text_to_instance(query_id, other_id, position)
                score = sr.forward_on_instance(neg_instance)['pos_score']
                neg_scores.append(score)
                paper_score.append([other_id, score])
            position = position + 1

        ranked_above = sum(1 for x in neg_scores if x > pos_score)
        ranked_below = sum(1 for x in neg_scores if x < pos_score)
        # probably rare case of ties:
        ranked_above += sum(0.5 for x in neg_scores if x==pos_score)
        ranked_below += sum(0.5 for x in neg_scores if x==pos_score)

        correct_order = correct_order + ranked_below
        incorrect_order = incorrect_order + ranked_above
        ndcg = 1/math.log2(2+ ranked_above)
        mrr = mrr + 1/(1 + ranked_above)
        ndcg_numerator = ndcg_numerator + ndcg
        rprec = rprec + (1 if ranked_above==0 else 0)
        #adjusted versions of each metric weight examples by 1 / (propensity score),
        # need to keep track of both numerator and denominator in weighted terms:
        adj_demonimator = adj_demonimator + adj
        adj_ndcg_numerator = adj_ndcg_numerator + adj*ndcg
        adj_correct_order = adj_correct_order + adj*ranked_below
        adj_incorrect_order = adj_incorrect_order + adj*ranked_above
        adj_mrr = adj_mrr + adj*(1/(1+ranked_above))
        adj_rprec = adj_rprec + adj*(1 if ranked_above==0 else 0)
        paper_score.sort(key=(lambda x: x[1]), reverse=True)
        paper_ranking = [x for [x, y] in paper_score]
        dict = {}
        dict[query_id] = paper_ranking
        output.append(dict)

    metrics = {
        "accuracy": correct_order / (correct_order + incorrect_order),
        "ndcg": ndcg_numerator / len(clicks),
        "mrr": mrr / len(clicks),
        "Rprec/P@1": rprec / len(clicks),
        "Adj-accuracy": adj_correct_order / (adj_incorrect_order + adj_correct_order),
        "Adj-ndcg": adj_ndcg_numerator / adj_demonimator,
        "Adj-mrr": adj_mrr / adj_demonimator,
        "Adj-Rprec/P@1": adj_rprec / adj_demonimator
    }

    return metrics


def get_recomm_metrics(data_paths:DataPaths, embeddings_path, val_or_test='test', cuda_device=-1):
    """Run the recommendations task evaluation.

    Arguments:
        data_paths {scidocs.paths.DataPaths} -- A DataPaths objects that points to 
                                                all of the SciDocs files
        embeddings_path {str} -- Path to the embeddings jsonl

    Keyword Arguments:
        val_or_test {str} -- Whether to return metrics on validation set (to tune hyperparams)
                             or the test set (what's reported in SPECTER paper)
        cuda_evice {str} -- For the pytorch model -> which cuda device to use

    Returns:
        metrics {dict} -- adj-NDCG and adj-P@1 for the task.
    """
    assert val_or_test in ('val', 'test'), "The val_or_test parameter must be one of 'val' or 'test'"
    # TODO(dougd): return validation metrics of val_or_test == 'val'
    
    print('Loading recomm embeddings...')
    with open(embeddings_path, 'r') as f:
        line = json.loads(next(f))
        num_dims = len(line['embedding'])

    print('Running the recomm task...')
    config_path = data_paths.recomm_config
    os.environ['CUDA_DEVICE'] = str(cuda_device)
    os.environ['EMBEDDINGS_PATH'] = embeddings_path
    os.environ['EMBEDDINGS_DIM'] = str(num_dims)
    os.environ['TRAIN_PATH'] = data_paths.recomm_train
    os.environ['VALID_PATH'] = data_paths.recomm_val
    if val_or_test == 'test':
        os.environ['TEST_PATH'] = data_paths.recomm_test
    else:
        os.environ['TEST_PATH'] = data_paths.recomm_val
        os.environ['VALID_PATH'] = ""
    os.environ['PROP_SCORE_PATH'] = data_paths.recomm_propensity_scores
    os.environ['PAPER_METADATA_PATH'] = data_paths.paper_metadata_recomm
    os.environ['jsonlines_embedding_format'] = "true"
    serialization_dir = os.path.join(data_paths.base_path, "recomm-tmp")
    simpapers_model_path = os.path.join(serialization_dir, "model.tar.gz")
    shutil.rmtree(serialization_dir, ignore_errors=True)
    command = \
        ['allennlp',
         'train', config_path, '-s', serialization_dir,
         '--include-package', 'scidocs.recommender']
    subprocess.run(command)
    metrics = evaluate_ranking_performance(simpapers_model_path, data_paths.recomm_test if val_or_test=='test'
       else data_paths.recomm_val, int(cuda_device))
    return {'recomm': {
        'adj-NDCG': np.round(100 * float(metrics['Adj-ndcg']), 2), 
        'adj-P@1': np.round(100 * float(metrics['Adj-Rprec/P@1']), 2),
        }}
