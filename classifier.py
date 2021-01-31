import torch
import allennlp
import numpy as np
from typing import Any, Dict
from allennlp.nn import util
from allennlp.models import Model
from allennlp.data import Vocabulary
from components.wgan import POGGenerator
from allennlp.data import TextFieldTensors
from components.wgan import POGDiscriminator
from components.autoencoder import POGEncoder, POGDecoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules import Seq2VecEncoder, LSTMSeq2VecEncoder
from components.auxilliary_classifier import POGAuxilliaryClassifier


class POG(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            embedder: TokenEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            batch_size: int,
            embedding_dim: int,
            wbrun: Any
    ):
        super().__init__(vocab)
        num_labels = vocab.get_vocab_size("labels")
        log.debug(f"Labels: {num_labels}.")

        self.embedder = embedder
        self.seq2vec_encoder = seq2vec_encoder
        self.encoder = POGEncoder()
        self.decoder = POGDecoder()
        self.generator = POGGenerator()
        self.discriminator = POGDiscriminator()
        self.auxilliary_classifier = POGAuxilliaryClassifier(vocab)
        self.accuracy = CategoricalAccuracy()
        wbrun.watch(self.classifier, log=all)
        log.debug("Model init complete.")

    def forward(
            self,
            sentence: TextFieldTensors,
            label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # log.debug(f"Forward pass starting. Sentence Dict: {sentence!r}")
        output = dict()

        # BEGIN training encoder/decoder and AC.
        # Output Shape: (batch_size, embedding_dim)
        xi = self.embedder(sentence)
        zi = self.seq2vec_encoder(xi)
        # Below step combines step 6 and 7, since loss is computed
        # inside the decoder, and we do not convert latent code
        # back to a discrete set of tokens, for the
        # "continuous approximation approach" from the paper.
        xi_prime, output["l_rec"] = self.decoder(zi, xi)
        # assuming we have the label, do next step
        # TODO: Consider absent label case too.
        _, output["l_ac"] = self.auxilliary_classifier(
            xi_prime,
            label
        )
        # END encoder/decoder and AC.

        # BEGIN training discriminator.
        epsilon = np.random.normal(size=(
            self.batch_size,
            self.embedding_dim
        ))
        # --- EOD 1/26 ---

        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output['probs'] = probs

        # log.debug(f"Forward pass complete. Probabilities: {probs!r}")

        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
            """log.debug("Calling wandb.log")
            wandb.log({
                "loss": output['loss'],
                "accuracy": self.accuracy.get_metric(reset=False)
            })"""
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
