
from scidocs.paths import DataPaths
from scidocs.embeddings import load_embeddings_from_jsonl
from scidocs.classification import classify, get_X_y_for_classification
from scidocs.user_activity_and_citations import get_qrel_metrics, make_run_from_embeddings
from scidocs.simpaper_click_eval import get_simpaper_metrics


# embeddings for all (currently not used)
embeddings_path = '/net/s2-research/recommender/scidocs_v1/paper_metadata_embedded.jsonl'
# embeddings = load_embeddings_from_jsonl(embeddings_path)

# all of the data paths
data_paths = DataPaths()

# MAG
# mag_embeddings_path = '/net/s2-research/recommender/embeddings/cite1hop-no-venue-mag.jsonl'
# mag_embeddings = load_embeddings_from_jsonl(mag_embeddings_path)
# X, y = get_X_y_for_classification(mag_embeddings, data_paths.mag_train, data_paths.mag_val, data_paths.mag_test)
# # if the algorithm that procued `embeddings` has nothing to tune we use:
# mag_test_f1 = classify(X['train'], y['train'], X['test'], y['test'])
# # if it DOES have something to tune, then you can tune it without leakage:
# mag_val_f1 = classify(X['train'], y['train'], X['val'], y['val'])
#
# # MeSH
# mesh_embeddings_path = '/net/s2-research/recommender/embeddings/cite1hop-no-venue-mesh.jsonl'
# mesh_embeddings = load_embeddings_from_jsonl(mesh_embeddings_path)
# X, y = get_X_y_for_classification(mesh_embeddings, data_paths.mesh_train, data_paths.mesh_val, data_paths.mesh_test)
# # if the algorithm that procued `embeddings` has nothing to tune we use
# mesh_test_f1 = classify(X['train'], y['train'], X['test'], y['test'])
#
#
# # write out run for a specific qrel
# # again, the validation file is only here in case you are tuning a baseline
# embeddings_path = '/net/s2-research/recommender/embeddings/cite1hop-no-venue-qrel-all.jsonl'
# embeddings_qrel = load_embeddings_from_jsonl(embeddings_path)
run_path = '/tmp/temp.run'
#
# make_run_from_embeddings(data_paths.coview_test, embeddings_qrel, run_path, topk=5, generate_random_embeddings=False)
# coview_results = get_qrel_metrics(data_paths.coview_test, run_path, metrics=('ndcg', 'map'))
#
# make_run_from_embeddings(data_paths.coread_test, embeddings_qrel, run_path, topk=5, generate_random_embeddings=False)
# coread_results = get_qrel_metrics(data_paths.coread_test, run_path, metrics=('ndcg', 'map'))
#
# make_run_from_embeddings(data_paths.cite_test, embeddings_qrel, run_path, topk=5, generate_random_embeddings=False)
# cite_results = get_qrel_metrics(data_paths.cite_test, run_path, metrics=('ndcg', 'map'))
#
# make_run_from_embeddings(data_paths.cocite_test, embeddings_qrel, run_path, topk=5, generate_random_embeddings=False)
# cocite_results = get_qrel_metrics(data_paths.cocite_test, run_path, metrics=('ndcg', 'map'))


#TODO read dims from embeddings
simpaper_results = get_simpaper_metrics(data_paths.recomm_test, data_paths.recomm_config, embeddings_path, run_path, "-1", "768")

print(simpaper_results)

# print(coview_results, coread_results, cite_results, cocite_results, simpaper_results)


# simpapers code goes below when it's done!

