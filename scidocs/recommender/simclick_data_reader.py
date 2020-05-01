""" Data reader for similar paper click data. """
from typing import Dict, List, Optional
import json
import logging

import jsonlines

from scidocs.recommender.simclick_triplet_sampler import SimClickTripletSampler
import numpy

import torch
from allennlp.data import Field
from overrides import overrides
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("simclick_data_reader")
class SimClickDataReader(DatasetReader):

    BLANK = '<blank>'

    def __init__(self,
                 lazy: bool = False,
                 paper_features_path: str = None,
                 paper_embeddings_path: str = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_results_per_query: int = 5,
                 jsonlines_embedding_format: bool = False
                 ) -> None:
        """

        Args:
            lazy: if false returns a list
            paper_features_path: path to the paper features json file (output by scripts.generate_paper_features.py)
            paper_embeddings_path: path to a file of paper embeddings
            tokenizer: tokenizer to be used for tokenizing strings
            token_indexers: token indexer for indexing vocab
            jsonlines_embedding_format: if the embeddings file is in jsonlines format, load it into a json
        """
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._token_indexer_venue = {"tokens": SingleIdTokenIndexer(namespace='venue')}
        with open(paper_features_path) as f_in:
            self.papers = json.load(f_in)
        if jsonlines_embedding_format:  # each line is a jsondict
            with jsonlines.open(paper_embeddings_path) as f_in:
                self.paper_embeddings = {e['paper_id']: e['embedding'] for e in f_in}
        else:
            with open(paper_embeddings_path) as f_in:
                self.paper_embeddings = json.load(f_in)
        self.embedding_dims = len(next(iter(self.paper_embeddings.values())))
        self.sdr = SimClickTripletSampler(max_results_per_query)


    @overrides
    def _read(self, simpaper_click_file: str):
        """
        Args:
            simpaper_click_file: path to the similar papers click file (output by similar_papers.py)
        """
        #todo: enable lazy read
        if simpaper_click_file:
            logger.info(f'reading contents of the file at: {simpaper_click_file}')
            with open(simpaper_click_file) as f_in:
                self.clicks = f_in.read().splitlines()
            logger.info(f'reading complete. Total {len(self.clicks)} records found.')
        else:
            self.clicks = None
        self.clicks = self.clicks[1:] #drop header row

        for line in self.clicks:
            # triplets are in format (p0, (p1, 1), (p2, 0))
            for triplet in self.sdr.generate_triplets(line):
                source_paper = triplet[0]
                pos_paper = triplet[1][0]
                neg_paper = triplet[2][0]
                pos_position = triplet[1][1]
                neg_position = triplet[2][1]
                yield (self.text_to_instance(source_paper, pos_paper, pos_position, neg_paper, neg_position))
        logger.info('done reading triplets')

    def jaccard(self, a:List, b:List):
        if len(set(a).union(set(b)))==0:
            return 0
        return len(set(a).intersection(set(b)))/len(set(a).union(set(b)))

    def x_overlap(self, source_paper, candidate_paper, x:str) -> float:
        if not source_paper in self.papers or not candidate_paper in self.papers:
            return 0
        source_x = self.papers[source_paper][x]
        cand_x = self.papers[candidate_paper][x]
        jac = self.jaccard(source_x, cand_x)
        return jac

    def author_match(self, source_paper, candidate_paper) -> float:
        return self.x_overlap(source_paper, candidate_paper, 'authors')

    def reference_overlap(self, source_paper, candidate_paper) -> float:
        return self.x_overlap(source_paper, candidate_paper, 'references')

    def citation_overlap(self, source_paper, candidate_paper) -> float:
        return self.x_overlap(source_paper, candidate_paper, 'cited_by')

    #returns whether the candidate cites source.  To be safe, checks both the references of the candidate,
    # and the citations of the source.  If either one indicates that candidate cites source, function returns 1
    # otherwise 0.
    def candidate_cites_source(self, source_paper, candidate_paper):
        source_cited_flag = source_paper in self.papers and candidate_paper in self.papers[source_paper]['cited_by']
        candidate_cites_flag = candidate_paper in self.papers and source_paper in self.papers[candidate_paper]['references']
        return int(source_cited_flag or candidate_cites_flag)


    @staticmethod
    def getHandleMissing(d: Dict[str, Dict[str, str]], key1: str, key2: str):
        if not d:
            return SimClickDataReader.BLANK
        if not key1 in d:
            s = SimClickDataReader.BLANK
        else:
            if not key2 in d[key1]:
                s = SimClickDataReader.BLANK
            else:
                s = d[key1][key2]
        if len(s)==0:
            return SimClickDataReader.BLANK
        else:
            return s

    def title_match(self, a:List, b:List):
        if len(a)==1 and len(b)==1 and a[0]==self.BLANK and b[0]==self.BLANK:
            return 0
        else:
            return self.jaccard([x.text for x in a], [x.text for x in b])

    def oldness(self, candidate_paper):
        if candidate_paper in self.papers:
            if self.papers[candidate_paper]['year'] is None:
                return 0
            try:
                fcand = float(self.papers[candidate_paper]['year'])
                return max(-1.0, min(1.0, (2020.0 - fcand)/20.0))
            except ValueError:
                return 0
        else:
            return 0

    def relative_oldness(self, source_paper, candidate_paper):
        if not source_paper in self.papers or not candidate_paper in self.papers:
            return 0
        elif self.papers[source_paper]['year'] is None or self.papers[candidate_paper]['year'] is None:
            return 0
        try:
            fsource = float(self.papers[source_paper]['year'])
            fcandidate = float(self.papers[candidate_paper]['year'])
            return max(-1.0, min(1.0, (fsource - fcandidate)/20.0))
        except ValueError:
            return 0

    def num_citations(self, candidate_paper):
        if not candidate_paper in self.papers:
            return 0
        else:
            return numpy.log(len(set(self.papers[candidate_paper]['cited_by']))+1)/10.0

    @staticmethod
    def valueOrZeros(d: Dict[str, list], k: str, num_dims: int):
        if not k in d:
            return numpy.zeros(num_dims)
        elif len(d[k])==0: #other failure mode
            return numpy.zeros(num_dims)
        else:
            return d[k]

    @overrides
    def text_to_instance(self,
                         source_paper: str,
                         positive_paper: str,
                         positive_position: int,
                         negative_paper: Optional[str] = None,
                         negative_position: Optional[int] = 0) -> Instance:
        fields: Dict[str, Field] = {}

        #todo: write a function that adds all fields for a paper, rather than duplicate lines as below
        #      also handle missing metadata more gracefully
        fields['pos_position'] = ArrayField(numpy.array([positive_position / 10]))
        fields['pos_author_match'] = ArrayField(numpy.array([self.author_match(source_paper, positive_paper)]))
        fields['pos_citation_overlap'] = ArrayField(numpy.array([self.citation_overlap(source_paper, positive_paper)]))
        fields['pos_reference_overlap'] = ArrayField(numpy.array([self.reference_overlap(source_paper, positive_paper)]))
        fields['pos_cites_query'] = ArrayField(numpy.array([self.candidate_cites_source(source_paper, positive_paper)]))
        fields['query_cites_pos'] = ArrayField(numpy.array([self.candidate_cites_source(positive_paper, source_paper)]))
        fields['pos_oldness'] = ArrayField(numpy.array([self.oldness(positive_paper)]))
        fields['pos_relative_oldness'] = ArrayField(numpy.array([self.relative_oldness(source_paper, positive_paper)]))
        fields['pos_number_citations'] = ArrayField(numpy.array([self.num_citations(positive_paper)]))


        (query_title, pos_title, neg_title) = \
            (self._tokenizer.tokenize(self.getHandleMissing(self.papers, x, 'title')) for x in (source_paper, positive_paper, negative_paper))

        fields['query_title'] = TextField(query_title,self._token_indexers)
        fields['pos_title'] = TextField(pos_title,self._token_indexers)
        fields['pos_title_match'] = ArrayField(numpy.array([self.title_match(query_title, pos_title)]))
        fields['query_emb'] = ArrayField(numpy.array(self.valueOrZeros(self.paper_embeddings, source_paper, self.embedding_dims)))
        fields['pos_emb'] = ArrayField(numpy.array(self.valueOrZeros(self.paper_embeddings, positive_paper, self.embedding_dims)))
        fields['pos_position'] = ArrayField(numpy.array([0]))

        if negative_paper:
            fields['neg_author_match'] = ArrayField(numpy.array([self.author_match(source_paper, negative_paper)]))
            fields['neg_citation_overlap'] = ArrayField(numpy.array([self.citation_overlap(source_paper, negative_paper)]))
            fields['neg_reference_overlap'] = ArrayField(numpy.array([self.reference_overlap(source_paper, negative_paper)]))
            fields['neg_cites_query'] = ArrayField(numpy.array([self.candidate_cites_source(source_paper, negative_paper)]))
            fields['query_cites_neg'] = ArrayField(numpy.array([self.candidate_cites_source(negative_paper, source_paper)]))
            fields['neg_oldness'] = ArrayField(numpy.array([self.oldness(negative_paper)]))
            fields['neg_relative_oldness'] = ArrayField(
                numpy.array([self.relative_oldness(source_paper, negative_paper)]))
            fields['neg_number_citations'] = ArrayField(numpy.array([self.num_citations(negative_paper)]))

            fields['neg_position'] = ArrayField(numpy.array([negative_position / 10]))
            fields['neg_title'] = TextField(neg_title,self._token_indexers)
            fields['neg_title_match'] = ArrayField(numpy.array([self.title_match(query_title, neg_title)]))
            fields['neg_emb'] = ArrayField(numpy.array(self.valueOrZeros(self.paper_embeddings, negative_paper, self.embedding_dims)))
            fields['neg_position'] = ArrayField(numpy.array([0]))

        return Instance(fields)
