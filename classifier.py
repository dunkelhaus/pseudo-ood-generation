import torch
import allennlp
import numpy as np
from typing import Any, Dict
from allennlp.nn import util
from allennlp.models import Model
from allennlp.data import Vocabulary
from components.encoder import POGEncoder
from components.decoder import POGDecoder
from allennlp.data import TextFieldTensors
from components.discriminator import POGDiscriminator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules import Seq2VecEncoder, LSTMSeq2VecEncoder
from components.auxilliary_classifier import POGAuxilliaryClassifier


class POG(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            embedder: TokenEmbedder,
            encoder: Seq2VecEncoder,
            wbrun: Any
    ):
        super().__init__(vocab)
        num_labels = vocab.get_vocab_size("labels")
        log.debug(f"Labels: {num_labels}.")

        self.embedder = embedder
        self.encoder = POGEncoder()
        self.decoder = POGDecoder()
        self.generator = POGGenerator()
        self.discriminator = POGDiscriminator()
        self.auxilliary_classifier = POGAuxilliaryClassifier()
        self.accuracy = CategoricalAccuracy()
        wbrun.watch(self.classifier, log=all)
        log.debug("Model init complete.")

    def forward(
            self,
            sentence: TextFieldTensors,
            label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        # log.debug(f"Forward pass starting. Sentence Dict: {sentence!r}")

        xi = self.embedder(sentence)
        zi = self.encoder(xi)
        zi = zi + np.random.normal(size=zi.shape)
        xi_prime = self.decoder(zi) # assuming it implements l_rec
        # TODO Step 7 here
        


        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {'probs': probs}

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
