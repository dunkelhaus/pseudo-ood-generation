import torch
import allennlp
import numpy as np
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules import Seq2VecEncoder, LSTMSeq2VecEncoder


class POGDecoder(LSTMSeq2VecEncoder):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            embedder: TokenEmbedder,
            num_layers: int = 1,
            bias: bool = True,
            dropout: float = 0.0,
            bidirectional: bool = False
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.embedder = embedder

    def forward(
            self,
            sentence: TextFieldTensors,
            hidden: Any
    ) -> Dict[str, torch.Tensor]:
        # Step 2
        _xi = self.embedder(sentence)
        xi = _xi
        xi, hidden = self.module(xi, hidden)

        return xi, hidden
