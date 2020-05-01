
from scidocs.paths import DataPaths
from scidocs.embeddings import load_embeddings_from_jsonl
from scidocs.classification import classify, get_X_y_for_classification
from scidocs.user_activity_and_citations import get_view_cite_read_metrics
from scidocs.recomm_click_eval import get_recomm_metrics


# all of the data paths
data_paths = DataPaths()

# MAG
mag_embeddings_path = '/net/nfs.corp/s2-research/recommender/embeddings/cite1hop-no-venue-mag.jsonl'
mag_embeddings = load_embeddings_from_jsonl(mag_embeddings_path)
X, y = get_X_y_for_classification(mag_embeddings, data_paths.mag_train, data_paths.mag_val, data_paths.mag_test)
# if the algorithm that procued `embeddings` has nothing to tune we use:
mag_test_f1 = classify(X['train'], y['train'], X['test'], y['test'])
# if it DOES have something to tune, then you can tune it without leakage:
#mag_val_f1 = classify(X['train'], y['train'], X['val'], y['val'])

# MeSH
mesh_embeddings_path = '/net/nfs.corp/s2-research/recommender/embeddings/cite1hop-no-venue-mesh.jsonl'
mesh_embeddings = load_embeddings_from_jsonl(mesh_embeddings_path)
X, y = get_X_y_for_classification(mesh_embeddings, data_paths.mesh_train, data_paths.mesh_val, data_paths.mesh_test)
mesh_test_f1 = classify(X['train'], y['train'], X['test'], y['test'])

# write out run for a specific qrel
# again, the validation file is only here in case you are tuning a baseline
embeddings_path = '/net/nfs.corp/s2-research/recommender/embeddings/cite1hop-no-venue-qrel-all.jsonl'
view_cite_read_metrics = get_view_cite_read_metrics(data_paths, embeddings_path=embeddings_path)

# recomm task
embeddings_path = '/net/nfs.corp/s2-research/recommender/embeddings/cite1hop-no-venue-simpapers.jsonl'
CUDA_DEVICE = -1
recomm_results = get_recomm_metrics(data_paths, embeddings_path, CUDA_DEVICE)

print(mag_test_f1, mesh_test_f1, view_cite_read_metrics, recomm_results)

