"""Explicit POG phases with isolated optimizers and local diagnostics."""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F
from wgan_gp.training import critic_step, frozen_parameters, generator_step

from .components.sentence_utils import annealed_temperature


@dataclass(frozen=True)
class POGSettings:
    """Engineering defaults, not settings claimed by the paper."""

    mode: str = "pog"
    batch_size: int = 32
    epochs: int = 80
    ae_lr: float = 1e-3
    auxiliary_lr: float = 1e-3
    generator_lr: float = 1e-4
    critic_lr: float = 1e-4
    gan_betas: tuple = (0.0, 0.9)
    gp_weight: float = 10.0
    entropy_weight: float = 1.0
    ae_clip_norm: float = 5.0
    critic_iterations: int = 5
    dmix_reconstruction_iterations: int = 1
    dmix_critic_iterations: int = 1
    max_steps: int = 63
    temperature_start: float = 1.0
    temperature_floor: float = 0.1
    baseline_decay: float = 0.9

    def __post_init__(self):
        if self.mode not in {"pog", "aepog", "pog-no-noise", "pog-no-soft-token"}:
            raise ValueError("Unknown POG mode")
        for name in ("batch_size", "epochs", "critic_iterations", "max_steps",
                     "dmix_reconstruction_iterations", "dmix_critic_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("ae_lr", "auxiliary_lr", "generator_lr", "critic_lr", "gp_weight",
                     "ae_clip_norm", "temperature_start", "temperature_floor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.entropy_weight) or self.entropy_weight < 0:
            raise ValueError("entropy_weight must be finite and nonnegative")
        if self.temperature_floor > self.temperature_start:
            raise ValueError("Temperature floor exceeds start")
        if not math.isfinite(self.baseline_decay) or not 0 <= self.baseline_decay < 1:
            raise ValueError("baseline_decay must be in [0, 1)")
        if len(self.gan_betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1
                                          for b in self.gan_betas):
            raise ValueError("Adam betas must be in [0, 1)")

    @property
    def reconstruction_noise(self):
        return self.mode != "pog-no-noise"

    @property
    def soft_tokens(self):
        return self.mode != "pog-no-soft-token"


def _finite(value, name):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"Non-finite {name}")


def negative_entropy(logits):
    """Return one negative predictive entropy per example."""
    log_probs = logits.log_softmax(-1)
    return (log_probs.exp() * log_probs).sum(-1)


class PseudoOODGenerationTrainer:
    """Run single phases or cycles; the caller owns loaders, RNGs, and files."""

    def __init__(self, model, settings=None, *, optimizers=None, log_callback=None):
        self.model, self.settings = model, settings or POGSettings()
        s = self.settings
        if optimizers is None:
            optimizers = {
                "ae": torch.optim.Adam(model.autoencoder.parameters(), lr=s.ae_lr),
                "auxiliary": torch.optim.Adam(model.auxiliary.parameters(), lr=s.auxiliary_lr),
                "generator": torch.optim.Adam(model.generator.parameters(), lr=s.generator_lr,
                                               betas=s.gan_betas),
                "critic": torch.optim.Adam(model.critic.parameters(), lr=s.critic_lr,
                                            betas=s.gan_betas),
            }
        if set(optimizers) != set(model.phase_modules):
            raise ValueError("Expected ae, auxiliary, generator, and critic optimizers")
        for name, module in model.phase_modules.items():
            expected = {id(p) for p in module.parameters() if p.requires_grad}
            actual = [id(p) for group in optimizers[name].param_groups for p in group["params"]]
            if not expected or set(actual) != expected or len(actual) != len(set(actual)):
                raise ValueError(f"Incorrect optimizer ownership: {name}")
        self.optimizers = optimizers
        self.log_callback = log_callback
        self.epoch = 0
        self.baseline = 0.0
        self.baseline_updates = 0
        self.update_counts, self.sample_counts = {}, {}

    @property
    def temperature(self):
        s = self.settings
        return annealed_temperature(self.epoch, max(s.epochs - 1, 1),
                                    s.temperature_start, s.temperature_floor)

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < self.settings.epochs:
            raise ValueError("epoch is a zero-based configured epoch")
        self.epoch = epoch

    @contextmanager
    def _phase(self):
        self.model.zero_grad(set_to_none=True)
        try:
            for p in self.model.parameters():
                _finite(p, "model parameters")
            yield
        finally:
            # Discard AE gradients from auxiliary CE and all stale owner gradients.
            self.model.zero_grad(set_to_none=True)

    def _step(self, loss, owner, clip=None):
        _finite(loss, "phase loss")
        loss.backward()
        parameters = [p for p in self.model.phase_modules[owner].parameters() if p.grad is not None]
        if not parameters:
            raise RuntimeError("Phase produced no owner gradients")
        for p in self.model.parameters():
            if p.grad is not None:
                _finite(p.grad, "phase gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf") if clip is None else clip,
                                              error_if_nonfinite=True)
        self.optimizers[owner].step()
        for p in self.model.phase_modules[owner].parameters():
            _finite(p, "updated parameters")
        return norm.item()

    def _record(self, phase, batch_size, result):
        if any(not math.isfinite(float(v)) for v in result.values()):
            raise FloatingPointError("Non-finite phase diagnostics")
        self.update_counts[phase] = self.update_counts.get(phase, 0) + 1
        self.sample_counts[phase] = self.sample_counts.get(phase, 0) + batch_size
        record = {"phase": phase, "epoch": self.epoch, "temperature": self.temperature,
                  "batch_size": batch_size, "updates": self.update_counts[phase],
                  "samples": self.sample_counts[phase], **result}
        if self.log_callback is not None:
            self.log_callback(record)
        return record

    def _dmix(self, dmix):
        if dmix and self.settings.mode != "aepog":
            raise ValueError("Dmix phases are exclusive to AEPOG")

    def reconstruction_step(self, tokens, *, dmix=False, rng=None):
        self._dmix(dmix)
        with self._phase():
            output = self.model.autoencoder(tokens, add_noise=self.settings.reconstruction_noise,
                                            temperature=self.temperature, generator=rng)
            norm = self._step(output["loss"], "ae", self.settings.ae_clip_norm)
            result = {"reconstruction_loss": output["loss"].item(), "gradient_norm": norm}
        return self._record("dmix_reconstruction" if dmix else "reconstruction", tokens.shape[0], result)

    def auxiliary_step(self, tokens, labels, *, rng=None):
        with self._phase():
            output = self.model.autoencoder(tokens, add_noise=self.settings.reconstruction_noise,
                                            temperature=self.temperature, generator=rng)
            if self.settings.soft_tokens:
                logits = self.model.auxiliary.forward_probabilities(
                    output["probabilities"], output["mask"], self.model.token_mapping)
            else:
                probs = output["probabilities"].detach()
                sampled = torch.multinomial(probs.flatten(0, 1), 1, generator=rng).view(tokens.shape[0], -1)
                logits = self.model.auxiliary(self.model.token_mapping[sampled], output["mask"])
            loss = F.cross_entropy(logits, labels)
            norm = self._step(loss, "auxiliary")
            result = {"auxiliary_loss": loss.item(), "gradient_norm": norm}
        return self._record("auxiliary", tokens.shape[0], result)

    def critic_step(self, tokens, *, dmix=False, noise=None, rng=None):
        self._dmix(dmix)
        with self._phase():
            with torch.no_grad():
                real = self.model.autoencoder.encode(tokens).detach()
            result = critic_step(self.model.generator, self.model.critic, self.optimizers["critic"],
                                 real, gp_weight=self.settings.gp_weight, noise=noise, rng=rng)
        return self._record("dmix_critic" if dmix else "critic", tokens.shape[0], result)

    def generator_wasserstein_step(self, batch_size, *, noise=None, rng=None):
        with self._phase():
            result = generator_step(self.model.generator, self.model.critic,
                                    self.optimizers["generator"], batch_size, noise=noise, rng=rng)
        return self._record("generator_wasserstein", batch_size, result)

    def generator_entropy_step(self, batch_size, *, noise=None, rng=None):
        m, s = self.model, self.settings
        with self._phase(), ExitStack() as stack:
            for module in (m.autoencoder, m.auxiliary, m.critic):
                stack.enter_context(frozen_parameters(module))
            codes = m.generator(m.generator.sample_latent(batch_size, rng=rng)
                                if noise is None else noise.detach())
            if codes.shape[0] != batch_size:
                raise ValueError("Noise batch size must match batch_size")
            rollout = m.autoencoder.rollout(codes, max_steps=s.max_steps, temperature=self.temperature,
                                           mode="soft" if s.soft_tokens else "sample", generator=rng)
            if s.soft_tokens:
                logits = m.auxiliary.forward_probabilities(rollout["probabilities"], rollout["mask"],
                                                            m.token_mapping)
                entropy = -negative_entropy(logits)
                loss = -s.entropy_weight * entropy.mean()
            else:
                with torch.no_grad():
                    logits = m.auxiliary(m.token_mapping[rollout["tokens"]], rollout["mask"])
                    entropy = -negative_entropy(logits)
                advantage = entropy.detach() - self.baseline
                loss = -s.entropy_weight * (advantage * rollout["sequence_log_probs"]).mean()
            norm = self._step(loss, "generator")
            result = {"entropy_loss": loss.item(), "entropy": entropy.mean().item(),
                      "gradient_norm": norm, "reward_baseline": self.baseline}
            if not s.soft_tokens:
                self.baseline = s.baseline_decay * self.baseline + (1 - s.baseline_decay) * entropy.mean().item()
                self.baseline_updates += 1
        return self._record("generator_entropy", batch_size, result)

    def train_cycle(self, tokens, labels, *, dmix_tokens=None, rng=None):
        """Reuse each supplied batch for configured repeats; draw fresh noise per step."""
        if self.settings.mode == "aepog" and dmix_tokens is None:
            raise ValueError("AEPOG requires an unlabeled Dmix token tensor")
        if self.settings.mode != "aepog" and dmix_tokens is not None:
            raise ValueError("POG and its ablations must not consume Dmix")
        # Reject labeled mappings before any phase updates.
        if dmix_tokens is not None and not isinstance(dmix_tokens, torch.Tensor):
            raise ValueError("Dmix accepts only an unlabeled token tensor")
        records = [self.reconstruction_step(tokens, rng=rng), self.auxiliary_step(tokens, labels, rng=rng)]
        records.extend(self.critic_step(tokens, rng=rng) for _ in range(self.settings.critic_iterations))
        records.append(self.generator_wasserstein_step(tokens.shape[0], rng=rng))
        records.append(self.generator_entropy_step(tokens.shape[0], rng=rng))
        if dmix_tokens is not None:
            records.extend(self.reconstruction_step(dmix_tokens, dmix=True, rng=rng)
                           for _ in range(self.settings.dmix_reconstruction_iterations))
            records.extend(self.critic_step(dmix_tokens, dmix=True, rng=rng)
                           for _ in range(self.settings.dmix_critic_iterations))
        return records

    def state_dict(self):
        """Trainer-only state; caller separately saves model, RNG, and loader states."""
        return {"settings": asdict(self.settings), "epoch": self.epoch,
                "baseline": self.baseline, "baseline_updates": self.baseline_updates,
                "update_counts": dict(self.update_counts), "sample_counts": dict(self.sample_counts),
                "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()}}

    def load_state_dict(self, state):
        if state["settings"] != asdict(self.settings):
            raise ValueError("Incompatible POG trainer settings")
        if not math.isfinite(state["baseline"]):
            raise ValueError("Non-finite policy baseline")
        self.set_epoch(state["epoch"])
        self.baseline, self.baseline_updates = state["baseline"], state["baseline_updates"]
        self.update_counts, self.sample_counts = dict(state["update_counts"]), dict(state["sample_counts"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
