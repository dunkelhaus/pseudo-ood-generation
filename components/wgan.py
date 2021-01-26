import torch
import allennlp
import numpy as np
from typing import Any, Dict
from allennlp.modules import Seq2VecEncoder, LSTMSeq2VecEncoder


class POGGenerator(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            embedder: Embedder,
            encoder: Seq2VecEncoder,
            wbrun: Any
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = None # TODO Implement decoder.
        num_labels = vocab.get_vocab_size("labels")

    def forward(
            self,
            epsilon: np.array
    ) -> Dict[str, torch.Tensor]:

        return


class POGDiscriminator(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            encoder: Seq2VecEncoder,
            wbrun: Any
    ):
        super().__init__(vocab)
        self.encoder = encoder
        self.decoder = None # TODO Implement decoder.
        num_labels = vocab.get_vocab_size("labels")
        log.debug(f"Labels: {num_labels}.")

    def forward(
            self,
            sentence: torch.Tensor,
            label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Step 2
        xi = self.embedder(sentence)

        # Step 3
        zi = self.encoder(xi)

        # Step 4
        # TODO Implement encoder_dim.
        epsilon = np.random.normal(loc=0.0, scale=1.0, size=enc_dim)
        zi_noised = zi + epsilon

        # Step 5
