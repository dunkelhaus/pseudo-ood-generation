"""Construct supported sentence models without legacy or image dependencies."""

import torch
from wgan_gp.models import Critic, Generator

from .components.ae.pog_ae import POGAutoEncoder
from .components.auxilliary_classifier import POGAuxiliaryClassifier
from .components.sentence_utils import cuda_device
from .pog import POG
from .trainer import POGSettings, PseudoOODGenerationTrainer


def build_model(generator_vocab_size, auxiliary_vocab_size, token_mapping, *,
                embedding_dim=100, code_dim=100, noise_dim=100, decoder_hidden_dim=None,
                gan_hidden_dim=512, gan_layers=4, num_classes=150, out_channels=128,
                kernel_sizes=(2, 3, 4, 5), mlp_hidden_size=512, pad_id=0, bos_id=2,
                eos_id=3, noise_std=1.0, generator_embeddings=None, auxiliary_embeddings=None,
                device="cuda:0"):
    """Build independent AE/auxiliary tables and fork-owned generator/critic."""
    device = cuda_device(device)
    mapping = torch.as_tensor(token_mapping, dtype=torch.long, device=device)
    ae = POGAutoEncoder(generator_vocab_size, embedding_dim, code_dim, decoder_hidden_dim,
                        pad_id, bos_id, eos_id, noise_std, generator_embeddings, device)
    auxiliary = POGAuxiliaryClassifier(auxiliary_vocab_size, embedding_dim, num_classes,
                                       out_channels, kernel_sizes, mlp_hidden_size, pad_id,
                                       auxiliary_embeddings, device)
    generator = Generator(noise_dim, code_dim, gan_hidden_dim, gan_layers, device=device)
    critic = Critic(code_dim, gan_hidden_dim, gan_layers, device=device)
    return POG(ae, auxiliary, generator, critic, mapping)


def build_trainer(model, settings=None, *, optimizers=None, log_callback=None):
    return PseudoOODGenerationTrainer(model, settings or POGSettings(),
                                      optimizers=optimizers, log_callback=log_callback)
