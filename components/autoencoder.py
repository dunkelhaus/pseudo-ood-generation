import torch
from torch import nn
from typing import Any, Dict
from allennlp.nn import util
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.modules import Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders import TokenEmbedder


class POGEncoder(Model):
    """
    LSTM encoder.
    Adds noise too.
    """
    def __init__(
            self,
            vocab: Vocabulary,
            wbrun: Any
    ):
        super().__init__(vocab)
        self.lstm = nn.LSTM(768, 100)
        # wbrun.watch(self.classifier, log=all)

    def forward(
            self,
            example: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        output = dict()

        out = self.lstm(example)
        epsilon = torch.randn(size=out.shape)
        output["encoding"] = out + epsilon

        return output


class POGDecoder(Model):
    """
    LSTM decoder.
    Uses cross entropy loss.
    """
    def __init__(
            self,
            vocab: Vocabulary,
            wbrun: Any
    ):
        super().__init__(vocab)
        self.lstm = nn.LSTM(100, 768)
        self.accuracy = CategoricalAccuracy()
        # wbrun.watch(self.classifier, log=all)

    def forward(
            self,
            example: torch.Tensor,
            target: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        output = dict()

        output["probs"] = nn.functional.log_softmax(
            self.lstm(example)
        )
        output["loss"] = nn.functional.nll_loss(
            output["probs"],
            target
        )

        return output


class AutoEncoder(Model):
    """
    Trains above encoder and decoder.
    """
    def __init__(
            self,
            vocab: Vocabulary,
            embedder: TokenEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            wbrun: Any
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.seq2vec_encoder = seq2vec_encoder
        num_labels = vocab.get_vocab_size("labels")
        print(f"Label check in autoencoder init: {num_labels}.")

        self.encoder = POGEncoder(vocab=vocab, wbrun=wbrun)
        self.decoder = POGDecoder(vocab=vocab, wbrun=wbrun)
        self.accuracy = CategoricalAccuracy()
        # wbrun.watch(self.classifier, log=all)

    def forward(
            self,
            example: TextFieldTensors
    ) -> Dict[str, torch.Tensor]:
        """
        Allennlp model forward pass specification.
        Must specify a loss key in returned dictionary to be trained
        by an allennlp trainer.
        """
        # log.debug(f"Forward pass starting. Sentence Dict: {sentence!r}")
        output = dict()

        # BEGIN training encoder/decoder and AC.
        # Output Shape: (batch_size, embedding_dim)
        as_embedding_seq = self.embedder(example)
        as_embedding_vec = self.seq2vec_encoder(as_embedding_seq)

        encoder_output = self.encoder(as_embedding_vec)
        decoder_output = self.decoder(
            encoder_output["encoding"],
            as_embedding_vec
        )

        output["encoder_loss"] = encoder_output["loss"]
        output["decoder_loss"] = decoder_output["loss"]
        output["decoder_probs"] = decoder_output["probs"]
        self.accuracy(output["decoder_probs"], as_embedding_vec)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
