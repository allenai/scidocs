import argparse
import jsonlines
import math
import torch
import os
import subprocess

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models import archival

import json

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
        params['dataset_reader'].params['paper_features_path'] = paper_features_path_override # '/net/nfs.corp/s2-research/recommender/online_experiment/metadata.json'
    if not paper_embeddings_path_override is None:
        params['dataset_reader'].params['paper_embeddings_path'] = paper_embeddings_path_override # '/net/nfs.corp/s2-research/recommender/online_experiment/cite-1hop-finetune.jsonl'

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
        [session_id, dt, query_id, clicked_id, clicked_pdf, other_id_str] = line.split('\t')

        other_ids = other_id_str.strip().split(",")
        position = 1
        paper_score = [] #used for output
        clicked_position = other_ids.index(clicked_id) + 1
        pos_instance = dr.text_to_instance(query_id, clicked_id, clicked_position)
        pos_score = sr.forward_on_instance(pos_instance)['pos_score']
        # rank by embedding hack:
        # pos_score = torch.nn.functional.cosine_similarity(torch.Tensor(pos_instance['pos_emb'].array), torch.Tensor(pos_instance['query_emb'].array), dim=0)
        paper_score.append([clicked_id, pos_score])
        neg_scores = []
        adj = 1 / adjClickDistribution[clicked_position-1]
        for other_id in other_ids:
            if not other_id == clicked_id:
                neg_instance = dr.text_to_instance(query_id, other_id, position)
                score = sr.forward_on_instance(neg_instance)['pos_score']
                # rank by embedding hack:
                # score = torch.nn.functional.cosine_similarity(torch.Tensor(neg_instance['pos_emb'].array), torch.Tensor(neg_instance['query_emb'].array), dim=0)
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

def get_simpaper_metrics(data_dir, embeddings_path, run_dir, cuda_device):
   #train allennlp model on given embeddings, write to archive file in run path:
    os.environ['CUDA_DEVICE'] = cuda_device
    os.environ['EMBEDDINGS_PATH'] = embeddings_path
    os.environ['jsonlines_embedding_format'] = "true"
    config_path = os.path.join(data_dir, "config")
    command = \
        ['allennlp',
         'train', config_path, '-s', run_dir,
         '--include-package', 's2_recommender']
    subprocess.run(command)
    metrics = evaluate_ranking_performance(simpapers_model_path, config['simpapers_test'], int(cuda_device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('d', type=str, help='Model serialization directory')
    parser.add_argument('model_archive_path', help='Model archive path')
    parser.add_argument('test_data_path', help='Path to the test file')
    parser.add_argument('--output-rankings-path', help='Path to output rankings')
    parser.add_argument('--cuda-device', dest='cuda_device', default=0, type=int, help='cuda device')
    parser.add_argument('--paper-features-path-override', help='Path to paper features path, will override config')
    parser.add_argument('--paper-embeddings-path-override', help='Path to paper embeddings, will override config')

    args = parser.parse_args()

    metrics = evaluate_ranking_performance(args.model_archive_path, args.test_data_path, args.cuda_device, args.output_rankings_path,
                                           args.paper_features_path_override, args.paper_embeddings_path_override)

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

