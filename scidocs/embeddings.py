import json
import numpy as np
from tqdm import tqdm

def load_embeddings_from_jsonl(embeddings_path):
    """Load embeddings from a jsonl file.
    The file must have one embedding per line in JSON format.
    It must have two keys per line: `paper_id` and `embedding`

    Arguments:
        embeddings_path -- path to the embeddings file

    Returns:
        embeddings -- a dictionary where each key is the paper id
                                   and the value is a numpy array 
    """
    embeddings = {}
    with open(embeddings_path, 'r') as f:
        for line in tqdm(f, desc='reading embeddings from file...'):
            line_json = json.loads(line)
            embeddings[line_json['paper_id']] = np.array(line_json['embedding'])
    return embeddings
