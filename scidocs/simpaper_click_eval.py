import argparse
import jsonlines
import math
import torch
import os
import subprocess

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

def evaluate_ranking_performance(archive_path, test_data_path, cuda_device, output_ranking_file=None,
                                 paper_features_path_override=None, paper_embeddings_path_override=None):

    archive = archival.load_archive(archive_path, cuda_device=cuda_device)
    params = archive.config
    sr = archive.model

    propensity_score_path = params['model'].params['propensity_score_path']

    with open(propensity_score_path) as f_in:
        adjClickDistribution = torch.Tensor(json.load(f_in)['scores'])

    #adjust to sum to one:
    adjClickDistribution = adjClickDistribution * len(adjClickDistribution) / (
        sum(adjClickDistribution))

    if not paper_features_path_override is None:
        params['dataset_reader'].params['paper_features_path'] = paper_features_path_override
    if not paper_embeddings_path_override is None:
        params['dataset_reader'].params['paper_embeddings_path'] = paper_embeddings_path_override

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
    if not output_ranking_file is None:
        with jsonlines.open(output_ranking_file, mode="w") as writer:
            writer.write_all(output)
    return metrics

def get_simpaper_metrics(data_paths:DataPaths, embeddings_path, run_dir, cuda_device, num_dims):
   #train allennlp model on given embeddings, write to archive file in run path:
    config_path = data_paths.recomm_config
    os.environ['CUDA_DEVICE'] = cuda_device
    os.environ['EMBEDDINGS_PATH'] = embeddings_path
    os.environ['EMBEDDINGS_DIM'] = num_dims
    os.environ['TRAIN_PATH'] = data_paths.recomm_train
    os.environ['VALID_PATH'] = data_paths.recomm_val
    os.environ['TEST_PATH'] = data_paths.recomm_test
    os.environ['PROP_SCORE_PATH'] = data_paths.recomm_propensity_scores
    os.environ['PAPER_METADATA_PATH'] = data_paths.recomm_paper_metadata
    os.environ['jsonlines_embedding_format'] = "true"
    serialization_dir = os.path.join(run_dir, "recomm-tmp")
    simpapers_model_path = os.path.join(serialization_dir, "model.tar.gz")
    shutil.rmtree(serialization_dir, ignore_errors=True)
    command = \
        ['allennlp',
         'train', config_path, '-s', serialization_dir,
         '--include-package', 'scidocs.recommender']
    subprocess.run(command)
    metrics = evaluate_ranking_performance(simpapers_model_path, data_paths.recomm_test, int(cuda_device), num_dims)
    return metrics