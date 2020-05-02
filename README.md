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
conda install -y -q -c conda-forge numpy pandas scikit-learn jsonlines tqdm sklearn-contrib-lightning
pip install pytrec_eval awscli allennlp==0.9
python setup.py install
```

To obtain the data, run this command after the package is installed (from inside the `scidocs` folder):

`aws s3 sync --no-sign-request s3://ai2-s2-research-public/specter/scidocs/ data/`

## How to run SciDocs
To obtain SciDocs metrics, you must first embed each entry in the 3 metadata files:

- `data/paper_metadata_mag_mesh.json`
- `data/paper_metadata_view_cite_read.json`
- `data/paper_metadata_recomm.json`

The embeddings must then reside in `jsonl` files with one json entry embedding per line, which will look something like this:
`{"paper_id": "0dfb47e206c762d2f4caeb99fd9019ade78c2c98", "embedding": [-3, -6, 0, ..., 2]}`

TODO: include the SPECTOR embedding.jsonl files and use those in the example instead.

Once you have these 3 embedding files you can get all of the relevant metrics as follows:

```
from scidocs import get_scidocs_metrics
from scidocs.paths import DataPaths

# point to the data, which should be in scidocs/data by default
data_paths = DataPaths()

# point to the generated embeddings.jsonl
classification_embeddings_path = 'data/paper_embeddings_mag_mesh.jsonl'
user_activity_and_citations_embeddings_path = 'data/paper_embeddings_view_cite_read.jsonl'
recomm_embeddings_path = 'data/paper_embeddings_recomm.jsonl'

# now run the evaluation
scidocs_metrics = get_scidocs_metrics(data_paths,
                                      classification_embeddings_path,
                                      user_activity_and_citations_embeddings_path,
                                      recomm_embeddings_path)

print(scidocs_metrics)
```

## Wrapper

To use SciDocs from command line you can use the provided wrapper:

```
python scripts/run.py \
--cls /path/to/classification_embeddings.jsonl \
--coview /path/to/coview_coread_citation_embeddings.jsonl \
--recomm /path/to/recomm_embeddings.jsonl 
```

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
