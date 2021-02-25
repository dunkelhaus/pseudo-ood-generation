import logging.config
from typing import Any, Tuple
from allennlp.models import Model
from allennlp.data import DataLoader
from allennlp.data import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from pseudo_ood_generation.components.ae.pog_ae import POGAutoEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from oos_detect.models.builders import build_grad_desc_with_adam_trainer

# Logger setup.
log = logging.getLogger(__name__)


def pog_ae_builders(
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        lr: float,
        num_epochs: int,
        vocab: Vocabulary,
        wbrun: Any
) -> Tuple[Model, Trainer]:
    """
    Simple wrapper for both model-specific builder fns.

    :returns Model, Trainer: The Model & Trainer objects
                                respectively.
    """
    model = build_model(vocab, wbrun)

    trainer = build_grad_desc_with_adam_trainer(
        model,
        serialization_dir,
        train_loader,
        dev_loader,
        lr=lr,
        num_epochs=num_epochs,
        wbrun=wbrun
    )

    return model, trainer


def build_model(vocab: Vocabulary, wbrun: Any) -> Model:
    """
    Build the Model object, along with the embedder and encoder.

    :param vocab: The pre-instantiated vocabulary object.
    :return Model: The model object itself.
    """
    log.debug("Building the model.")
    # vocab_size = vocab.get_vocab_size("tokens")

    # TokenEmbedder object.
    # train_parameters is set to false to prevent updating params.
    bert_embedder = PretrainedTransformerEmbedder(
        "bert-base-uncased",
        train_parameters=False
    )

    # TextFieldEmbedder that wraps TokenEmbedder objects. Each
    # TokenEmbedder output from one TokenIndexer--the data produced
    # by a TextField is a dict {names:representations}, hence
    # TokenEmbedders have corresponding names.
    embedder: TextFieldEmbedder = BasicTextFieldEmbedder(
        {"tokens": bert_embedder}
    )

    log.debug("Embedder built.")
    # Have set requires_grad to false for the moment, to prevent
    # backpropping gradients to the linear layer in bert pooler.
    encoder = BertPooler("bert-base-uncased", requires_grad=False)
    # encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(768,20,batch_first=True))
    log.debug("Encoder built.")

    return POGAutoEncoder(vocab, embedder, encoder, wbrun).cuda(0)
