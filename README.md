# SciDocs - The Dataset Evaluation Suite for SPECTER

[**SPECTER Public API**](https://github.com/allenai/paper-embedding-public-apis) |
[**SPECTER Code Base**](https://github.com/allenai/specter) |
[**Paper**](https://arxiv.org/pdf/2004.07180.pdf) 

This repository contains code, link to data, and instructions to use the
[SciDocs](https://arxiv.org/pdf/2004.07180.pdf) evaluation suite.

## Installation
To install this package, run the following:

```bash
git clone https://github.com/allenai/scidocs.git
cd scidocs
conda create -y --name scidocs python==3.7
conda activate scidocs
conda install -y -q -c conda-forge numpy pandas scikit-learn=0.22.2 jsonlines tqdm sklearn-contrib-lightning pytorch
pip install pytrec_eval awscli allennlp==0.9 overrides==3.1.0
python setup.py install
```

To obtain the data, run this command after the package is installed (from inside the `scidocs` folder):  
```[Expected download size is: 4.6 GiB]```

`aws s3 sync --no-sign-request s3://ai2-s2-research-public/specter/scidocs/ data/`

Note: if you're having issues, make sure you're in the `us-west-2` region.

## Windows Support
Because `pytrec_eval` does not support Windows, you won't be able to install `SciDocs` on Windows. The data, however, is still accessible via the `awscli` command above.

## How to run SciDocs
To obtain SciDocs metrics, you must first embed each entry in the 3 metadata files:

- `data/paper_metadata_mag_mesh.json`
- `data/paper_metadata_view_cite_read.json`
- `data/paper_metadata_recomm.json`

The embeddings must then reside in `jsonl` files with one json entry embedding per line, which will look something like this:
`{"paper_id": "0dfb47e206c762d2f4caeb99fd9019ade78c2c98", "embedding": [-3, -6, 0, ..., 2]}`

We include the SPECTER embeddings as well. Here is how to reproduce the results in the SPECTER paper:

Once you have these 3 embedding files you can get all of the relevant metrics as follows:

```
from scidocs import get_scidocs_metrics
from scidocs.paths import DataPaths

# point to the data, which should be in scidocs/data by default
data_paths = DataPaths()

# point to the included embeddings jsonl
classification_embeddings_path = 'data/specter-embeddings/cls.jsonl'
user_activity_and_citations_embeddings_path = 'data/specter-embeddings/user-citation.jsonl'
recomm_embeddings_path = 'data/specter-embeddings/recomm.jsonl'

# now run the evaluation
scidocs_metrics = get_scidocs_metrics(
    data_paths,
    classification_embeddings_path,
    user_activity_and_citations_embeddings_path,
    recomm_embeddings_path,
    val_or_test='test',  # set to 'val' if tuning hyperparams
    n_jobs=12,  # the classification tasks can be parallelized
    cuda_device=-1  # the recomm task can use a GPU if this is set to 0, 1, etc
)

print(scidocs_metrics)
```
And you should see the following output:

`{'mag': {'f1': 81.95}, 'mesh': {'f1': 86.44}, 'co-view': {'map': 83.63, 'ndcg': 91.5}, 'co-read': {'map': 84.46, 'ndcg': 92.39}, 'cite': {'map': 88.3, 'ndcg': 94.88}, 'co-cite': {'map': 88.11, 'ndcg': 94.77}, 'recomm': {'adj-NDCG': 53.9, 'adj-P@1': 20.0}}`

Which matches exactly the last row of Table 1 in the SPECTER paper. Your results should be identical, with the exception of `recomm` due to a lack of reproducibility guarantees from PyTorch: https://pytorch.org/docs/stable/notes/randomness.html. 

To run your own models, you need to generate your own embedding jsonl files. To tune hyperparameters,
you can set the `val_or_test='val'` in the `get_scidocs_metrics` function and use the resulting values as part
of your objective function.

## Wrapper

To use SciDocs from command line you can use the provided wrapper:

```
python scripts/run.py \
--cls data/specter-embeddings/cls.jsonl \
--user-citation data/specter-embeddings/user-citation.jsonl \
--recomm data/specter-embeddings/recomm.jsonl \
--val_or_test test \
--n-jobs 12 \
--cuda-device -1
```

## MAG and MeSH Class Labels
The MAG and MeSH datasets included with this repo have integer labels starting at 0. Here is how these integers map to class names. 

For MeSH:
```
0 	 Cardiovascular diseases
1 	 Chronic kidney disease
2 	 Chronic respiratory diseases
3 	 Diabetes mellitus
4 	 Digestive diseases
5 	 HIV/AIDS
6 	 Hepatitis A/B/C/E
7 	 Mental disorders
8 	 Musculoskeletal disorders
9 	 Neoplasms (cancer)
10 	 Neurological disorders
```

And for MAG:
```
0	Art
1	Biology
2	Business
3	Chemistry
4	Computer science
5	Economics
6	Engineering
7	Environmental science
8	Geography
9	Geology
10	History
11	Materials science
12	Mathematics
13	Medicine
14	Philosophy
15	Physics
16	Political science
17	Psychology
18	Sociology
```

## Errata

The SPECTER paper erroneously lists the number of training examples in the recommendations task as 20K.  The correct value is 4K.

## Citation

Please cite the [SPECTER paper](https://arxiv.org/pdf/2004.07180.pdf) as:  

```
@inproceedings{specter2020cohan,
  title={SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  author={Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle={ACL},
  year={2020}
}
```
