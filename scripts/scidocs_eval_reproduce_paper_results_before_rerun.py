
from scidocs.paths import DataPaths
from scidocs.embeddings import load_embeddings_from_jsonl
from scidocs.classification import classify, get_X_y_for_classification
from scidocs.user_activity_and_citations import get_view_cite_read_metrics
from scidocs.recomm_click_eval import get_recomm_metrics


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mag', help='path to mag embeddings')
    parser.add_argument('--mesh', help='path to mesh embeddings')
    parser.add_argument('--coview', help='path to coview/coread/citation embeddings')
    parser.add_argument('--recomm', help='path to recommender related embeddings')
    parser.add_argument('--n-jobs', default=12, help='number of parallel jobs for classification (related to mesh/mag metrics)', type=int)
    parser.add_argument('--cuda-device', default=-1, help='if you want to use gpu for training the recommendation model')
    args = parser.parse_args()

    # all of the data paths
    data_paths = DataPaths()

    # MAG
    mag_embeddings_path = args.mag
    mag_embeddings = load_embeddings_from_jsonl(mag_embeddings_path)
    X, y = get_X_y_for_classification(mag_embeddings, data_paths.mag_train, data_paths.mag_val, data_paths.mag_test)
    # if the algorithm that procued `embeddings` has nothing to tune we use:
    mag_test_f1 = classify(X['train'], y['train'], X['test'], y['test'], n_jobs=args.n_jobs)
    # if it DOES have something to tune, then you can tune it without leakage:
    #mag_val_f1 = classify(X['train'], y['train'], X['val'], y['val'])
    print('mag', mag_test_f1)

    # MeSH
    mesh_embeddings_path = args.mesh
    mesh_embeddings = load_embeddings_from_jsonl(mesh_embeddings_path)
    X, y = get_X_y_for_classification(mesh_embeddings, data_paths.mesh_train, data_paths.mesh_val, data_paths.mesh_test)
    mesh_test_f1 = classify(X['train'], y['train'], X['test'], y['test'], n_jobs=args.n_jobs)
    print('mesh', mesh_test_f1)

    # write out run for a specific qrel
    # again, the validation file is only here in case you are tuning a baseline
    embeddings_path = args.coview
    view_cite_read_metrics = get_view_cite_read_metrics(data_paths, embeddings_path=embeddings_path)
    print(view_cite_read_metrics)

    # recomm task
    embeddings_path = args.recomm
    CUDA_DEVICE = args.cuda_device
    recomm_results = get_recomm_metrics(data_paths, embeddings_path, CUDA_DEVICE)

    print('mag f1:', mag_test_f1)
    print('mesh f1:', mesh_test_f1)
    print(view_cite_read_metrics)
    print(recomm_results)

if __name__ == '__main__':
    main()




