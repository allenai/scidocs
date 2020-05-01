# scidocs
Dataset accompanying the SPECTER model, as well as evaluation code.

To install this package, run the following:

```
git clone https://github.com/allenai/scidocs.git
cd scidocs
conda create -y --name scidocs python==3.7
conda activate scidocs
conda install -y -q -c conda-forge numpy pandas scikit-learn jsonlines tqdm sklearn-contrib-lightning
pip install pytrec_eval awscli allennlp==0.9
python setup.py install
```

To obtain the data, run this command after the package is installed:

`aws s3 sync --no-sign-request s3://ai2-s2-research-public/specter/scidocs/ data/`

## How To
To obtain SciDocs metrics, you must first embed each entry in the 3 metadata files:

- data/paper_metadata_mag_mesh.json
- data/paper_metadata_view_cite_read.json
- data/paper_metadata_recomm.json

The embeddings must then reside in `jsonl` files with one json entry embedding per line, which will look something like this:
`{"paper_id": "0dfb47e206c762d2f4caeb99fd9019ade78c2c98", "embedding": [-3, -6, 0, ..., 2]}`

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
```
