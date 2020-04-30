"""
The script reads a line of similar papers click data and returns it in triplet format
i.e., ['Query_paper', ('Positive_paper_id', 1), ('Negative_paper_id', 0)
"""
import operator
from typing import List, Tuple, Dict, Optional, Generator, NoReturn, Iterator

import numpy as np
import random
from sklearn.utils import check_random_state
import logging
import csv

logger = logging.getLogger(__file__)  # pylint: disable=invalid-name


class SimClickTripletSampler:

    def __init__(self,
                 max_samples_per_query: int,
                 random_state: Optional[int] = 12) -> NoReturn:
        self.max_samples_per_query = max_samples_per_query
        self.random_state_ = check_random_state(random_state)
        click_distribution = []

    def generate_triplets(self, line:str) -> Iterator[List[Tuple]]:
        """ Generate triplets from a line of similar papers click data

               This generates a list of triplets each query according to:
                   [(query_id, (positive_id, 1), (negative_id, 0)), ...]
               The upperbound of the list length is according to self.samples_per_query

               Args:
                   line: a single line from the output of similar_papers.py

               Returns:
                   Lists of triplet tuples
               """
        #
        [query_id, clicked_id, other_id_str] = list(csv.reader([line], delimiter=',', quotechar='"'))[0]
        other_ids = other_id_str.strip().split(",")
        position=1
        clicked_position = other_ids.index(clicked_id)+1
        count = 0
        for other_id in other_ids:
            if not other_id==clicked_id:
                yield (query_id, (clicked_id, clicked_position), (other_id, position))
                count = count + 1
            position = position + 1
            if count == self.max_samples_per_query:
                break
