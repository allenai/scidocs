
from scidocs.paths import DataPaths
from scidocs import get_scidocs_metrics

import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', '--classification-embeddings-path', dest='cls', help='path to classification related embeddings (mesh and mag)')
    parser.add_argument('--coview', '--user_activity_and_citations_embeddings_path', dest='coview', help='path to user activity embeddings (coview, copdf, cocite, citation)')
    parser.add_argument('--recomm', '--recomm_embeddings_path', dest='recomm', help='path to recommender related embeddings')
    parser.add_argument('--n-jobs', default=12, help='number of parallel jobs for classification (related to mesh/mag metrics)', type=int)
    parser.add_argument('--cuda-device', default=-1, help='if you want to use gpu for training the recommendation model')
    parser.add_argument('--data-path', default=None, help='path to the data directory where scidocs files reside. If None, it will default to the `data/` directory')
    args = parser.parse_args()

    data_paths = DataPaths(args.data_path)

    scidocs_metrics = get_scidocs_metrics(data_paths, args.cls, args.coview, args.recomm,
                                        n_jobs=args.n_jobs,
                                        cuda_device=args.cuda_device)

    print(json.dumps(scidocs_metrics, indent=2))

if __name__ == '__main__':
    main()




