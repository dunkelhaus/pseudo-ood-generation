import torch
from torch import nn
from allennlp.models import Model
from typing import Tuple, Any, Dict
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors


class POGAuxilliaryClassifier(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            wbrun: Any,
            in_channels: int = 768,
            out_channels: int = 128,
            kernel_sizes: Tuple[int] = (2, 3, 4, 5),
            mlp_hidden_size: int = 512
    ):
        super().__init__(vocab)
        num_labels = vocab.get_vocab_size("labels")
        print(f"Labels: {num_labels}.")

        # in_channels: the word embedding size (100d in paper).
        # out_channels: the number of feature maps (128 in paper).
        # kernel_size: kernel sizes ([2, 3, 4, 5] in paper).
        # Final kernel dims: kernel_size * in_channels * out_channels.
        self.cnn = tuple(torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros'
            ) for kernel_size in kernel_sizes
        )

        self.pool = tuple(
            torch.nn.MaxPool1d(
                kernel_size,
                stride=None,
                padding=0,
                dilation=1,
                return_indices=False,
                ceil_mode=False
            ) for kernel_size in kernel_sizes
        )

        in_features = out_channels * len(kernel_sizes)
        self.mlp1 = torch.nn.Linear(
            in_features,
            mlp_hidden_size,
            bias=True
        )
        self.mlp2 = torch.nn.Linear(
            mlp_hidden_size,
            mlp_hidden_size,
            bias=True
        )
        self.mlp3 = torch.nn.Linear(
            mlp_hidden_size,
            num_labels,
            bias=True
        )

    def forward(
            self,
            vector: torch.Tensor,
            label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Defines forward prop functions.

        :param sentence: Input sentence.
        """
        # TODO Apply CNN and MLP to output of embedder.
        # Embedder output should be sized (batch*max_sen_len*emb_dim).
        # max_sen_len is num_tokens in embedder output (allen visual).
        # Remember to use cross entropy loss first.
        # NOTE Remember to apply activations to each layer (keras diff).
        x = vector

        for i in range(len(self.kernel_sizes)):
            x = nn.functional.relu(self.cnn[i](x))

        x = nn.functional.relu(self.mlp1(x))
        x = nn.functional.relu(self.mlp2(x))
        x = self.mlp3(x)

        loss = nn.functional.cross_entropy(x, label)

        return x, loss
