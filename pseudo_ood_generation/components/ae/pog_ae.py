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
        self.lstm = nn.LSTM(768, 100, batch_first=True)
        # wbrun.watch(self.classifier, log=all)

    def forward(
            self,
            sentence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        output = dict()

        # print(f"Input to encooder's lstm: {sentence.shape}")
        out, cell_states = self.lstm(sentence)
        # print(out, out.shape)
        epsilon = torch.randn(size=out.shape).cuda(0)
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
        self.lstm = nn.LSTM(100, 768, batch_first=True)
        # wbrun.watch(self.classifier, log=all)

    def forward(
            self,
            sentence: torch.Tensor,
            label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        output = dict()

        out, cell_states = self.lstm(sentence)
        # print(f"Decoder output shape: {out.shape}")
        # print(f"Vocab size: {self.vocab.get_vocab_size()}")
        output["encoding"] = out

        return output


class POGAutoEncoder(Model):
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
            sentence: TextFieldTensors,
            label=None
    ) -> Dict[str, torch.Tensor]:
        """
        Allennlp model forward pass specification.
        Must specify a loss key in returned dictionary to be trained
        by an allennlp trainer.
        """
        # log.debug(f"Forward pass starting. Sentence Dict: {sentence!r}")
        output = dict()
        # print(sentence["tokens"]["token_ids"].shape)

        # Output Shape: (batch_size, num_tokens, embedding_dim)
        embedding_seq = self.embedder(sentence)
        # print(torch.max(embedding_seq), torch.min(embedding_seq))
        embedding_seq = torch.nn.functional.normalize(embedding_seq, dim=2)
        # print(torch.max(embedding_seq), torch.min(embedding_seq))
        # print(embedding_seq[0])

        # TODO: Try it without the mask here once.
        # mask = util.get_text_field_mask(sentence)
        # Output Shape: (batch_size, embedding_dim)
        # embedding_vec = self.seq2vec_encoder(
        #    embedding_seq,
        #    mask
        # )

        encoder_output = self.encoder(embedding_seq)
        output["encoder_out"] = encoder_output["encoding"]

        decoder_output = self.decoder(output["encoder_out"])

        output["probs"] = nn.functional.log_softmax(
            decoder_output["encoding"],
            dim=1
        )
        print(f"Output probs shape: {output['probs'].shape}")
        print(f"Output probs shape: {output['probs'].unsqueeze(-1).shape}")
        # print(f"Output probabilities: {output['probs']}")
        output["loss"] = nn.functional.binary_cross_entropy_with_logits(
            output["probs"],
            embedding_seq
        )

        # self.accuracy(output["probs"], embedding_seq)

        return output

"""    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}"""
