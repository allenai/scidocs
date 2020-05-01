from typing import Dict, Union, Optional

import torch

from torch import nn
from overrides import overrides

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, TimeDistributed, FeedForward
from allennlp.nn import RegularizerApplicator, InitializerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy, Average

import json

@Model.register('simpaper_recommender')
class SimpaperRecommender(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 feedforward: FeedForward,
                 ranking_loss_margin: float,
                 text_encoder: Seq2VecEncoder,
                 paper_embeddings_size: int,
                 encode_title: bool,
                 project_query: bool,
                 propensity_score_path: str,
                 paper_to_vec_dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.text_encoder = text_encoder

        check_dimensions_match(self.text_field_embedder.get_output_dim(), self.text_encoder.get_input_dim(),
                               "text embedder output dim", "text encoder input dim")

        if paper_to_vec_dropout:
            self.dropout = nn.Dropout(p=paper_to_vec_dropout)
        else:
            self.dropout = None

        self.project_query = project_query
        if project_query:
            self.query_projection = nn.Linear(paper_embeddings_size, paper_embeddings_size)


        self.ff_positive_score = feedforward
        self.encode_title = encode_title
        self.num_text_components_for_paper_to_vec = 1 if encode_title else 0

        self.num_extra_numeric_features = 11
        self.paper_embeddings_size = paper_embeddings_size
        self.total_paper_output_size = self.text_encoder.get_output_dim() * self.num_text_components_for_paper_to_vec
        self.expected_ff_input_dim = 2 * self.total_paper_output_size + self.num_extra_numeric_features
        check_dimensions_match(self.expected_ff_input_dim,
                               feedforward.get_input_dim(),
                               "expected feedforward input dim",
                               "actual feedforward input dim")
        self.loss = nn.MarginRankingLoss(margin=ranking_loss_margin, reduction='none')
        self.accuracy = CategoricalAccuracy()
        self.saved_loss = Average()

        with open(propensity_score_path) as f_in:
            adj_click_distribution = torch.Tensor(json.load(f_in)['scores'])

        # normalize so that average propensity score is equal to one (to keep loss on similar order as without adj):
        adj_click_distribution = adj_click_distribution * len(adj_click_distribution) / (sum(adj_click_distribution))

        self.register_buffer('adj_click_distribution', adj_click_distribution)

        initializer(self)

    def _paper_to_vec(self, title):
        """
        Input: (batch_size, num_tokens) for `title` and `abstract`
        Output: (batch_size, num_text_components_for_paper_to_vec * embedding_len)
        """
        batch_size = title.size(0)
        num_tokens_title = title.size(1)
        embedded_title = self.text_field_embedder({"tokens": title})
        check_dimensions_match(embedded_title.size(),
                               (batch_size, num_tokens_title, self.text_field_embedder.get_output_dim()),
                               "embedded title output dim", "embedded title expected output dim")

        mask_title = util.get_text_field_mask({"tokens": title})
        check_dimensions_match(mask_title.size(), (batch_size, num_tokens_title),
                               "mask title output dim", "mask title expected output dim")

        if self.dropout:
            embedded_title = self.dropout(embedded_title)
            check_dimensions_match(embedded_title.size(),
                                   (batch_size, num_tokens_title, self.text_field_embedder.get_output_dim()),
                                   "embedded title output dim after dropout", "embedded title expected output dim")

        encoded_title = self.text_encoder(embedded_title, mask_title)
        check_dimensions_match(encoded_title.size(), (batch_size, self.text_encoder.get_output_dim()),
                               "encoded title output dim", "encoded title expected output dim")

        if self.dropout:
            encoded_title = self.dropout(encoded_title)
            check_dimensions_match(encoded_title.size(),
                                   (batch_size, self.text_encoder.get_output_dim()),
                                   "encoded title output dim after dropout", "encoded title expected output dim")

        paper_vec = encoded_title
        return paper_vec


    @staticmethod
    def read_propensity_file(infile:str):
        with open(infile) as f_in:
            return json.load(f_in)

    @overrides
    def forward(self,
                query_title: torch.LongTensor,
                query_emb: torch.Tensor,
                pos_author_match: torch.Tensor,
                pos_citation_overlap: torch.Tensor,
                pos_reference_overlap: torch.Tensor,
                pos_cites_query: torch.Tensor,
                query_cites_pos: torch.Tensor,
                pos_oldness: torch.Tensor,
                pos_relative_oldness: torch.Tensor,
                pos_number_citations: torch.Tensor,
                pos_position: torch.Tensor,
                pos_title: Dict[str, torch.Tensor],
                pos_title_match: torch.Tensor,
                pos_emb: torch.Tensor,
                neg_author_match: torch.LongTensor = None,
                neg_citation_overlap: torch.Tensor = None,
                neg_reference_overlap: torch.Tensor = None,
                neg_cites_query: torch.Tensor = None,
                query_cites_neg: torch.Tensor = None,
                neg_oldness: torch.Tensor = None,
                neg_relative_oldness: torch.Tensor = None,
                neg_number_citations: torch.Tensor = None,
                neg_position: torch.LongTensor = None,
                neg_title: Dict[str, torch.Tensor] = None,
                neg_title_match: torch.Tensor = None,
                neg_emb: torch.Tensor = None):
        # query_title["tokens"] is (batch size x num tokens in title)
        batch_size = query_title["tokens"].size(0)
        if self.text_encoder and self.encode_title:
            query_paper_encoding = self._paper_to_vec(query_title["tokens"])
            check_dimensions_match(query_paper_encoding.size(), (batch_size, self.total_paper_output_size),
                                   "Query paper encoding size", "Expected paper encoding size")

            pos_paper_encoding = self._paper_to_vec(pos_title["tokens"])
            check_dimensions_match(pos_paper_encoding.size(), (batch_size, self.total_paper_output_size),
                                   "Positive paper encoding size", "Expected paper encoding size")
            if neg_title:
                # neg_paper_encoding is (batch size x size of embedding)
                neg_paper_encoding = self._paper_to_vec(neg_title["tokens"])
                check_dimensions_match(neg_paper_encoding.size(), (batch_size, self.total_paper_output_size),
                                       "Negative paper encoding size", "Expected paper encoding size")
        #pos_features holds additional features about this instance, is (batch size x num_extra_numeric_features)
        if self.project_query:
            proj_query_emb = self.query_projection(query_emb)
        else:
            proj_query_emb = query_emb

        pos_emb_sim = torch.nn.functional.cosine_similarity(proj_query_emb, pos_emb, dim=1)
        pos_emb_sim = pos_emb_sim.view(-1, 1)

        pos_features = torch.cat([pos_author_match, pos_position, pos_title_match, pos_emb_sim,
                                  pos_citation_overlap, pos_reference_overlap, pos_cites_query, query_cites_pos,
                                  pos_oldness, pos_relative_oldness, pos_number_citations],dim=1)
        check_dimensions_match(pos_features.size(), (batch_size, self.num_extra_numeric_features),
                               "Positive features size", "Expected positive features size")

        # positive_paper_score is (batch size x 1)
        if self.text_encoder and self.encode_title:
            positive_paper_score = self.ff_positive_score(
                torch.cat([query_paper_encoding, pos_paper_encoding, pos_features], dim=1))
        else:
            positive_paper_score = self.ff_positive_score(pos_features)
        check_dimensions_match(positive_paper_score.size(), (batch_size, 1),
                               "Positive score size", "Expected positive scoresize")
        if neg_title:
            # negative_paper_score is (batch size x 1)
            neg_emb_sim = torch.nn.functional.cosine_similarity(proj_query_emb, neg_emb, dim=1)
            neg_emb_sim = neg_emb_sim.view(-1, 1)

            neg_features = torch.cat([neg_author_match, neg_position, neg_title_match, neg_emb_sim,
                                      neg_citation_overlap, neg_reference_overlap, neg_cites_query, query_cites_neg,
                                      neg_oldness, neg_relative_oldness, neg_number_citations], dim=1)
            if self.text_encoder and self.encode_title:
                negative_paper_score = self.ff_positive_score(
                    torch.cat([query_paper_encoding, neg_paper_encoding, neg_features], dim=1))
            else:
                negative_paper_score = self.ff_positive_score(neg_features)
            check_dimensions_match(negative_paper_score.size(), (batch_size, 1),
                                   "negative score size", "Expected negative score size")
            long_pos_position = torch.round(pos_position*10-1).long()
            propensity_score = self.adj_click_distribution[long_pos_position]
            # loss is a batch size x 1 vector
            loss = self.loss(positive_paper_score, negative_paper_score,
                             torch.ones_like(positive_paper_score))
            loss = loss / propensity_score
            loss = torch.mean(loss)
            check_dimensions_match(loss.dim(), 0,
                                   "Loss size", "Expected loss size")

        output = {}
        output['pos_score'] = positive_paper_score
        if neg_title:
            self.accuracy(torch.cat([positive_paper_score, negative_paper_score], dim=1), torch.zeros(len(positive_paper_score)))
            self.saved_loss(loss.item()) #NOTE averages across batches, which is a bit wrong unless total examples is divisible by batch size
            output['neg_score'] = negative_paper_score
            output['loss'] = loss
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset),
                "loss": self.saved_loss.get_metric(reset)}