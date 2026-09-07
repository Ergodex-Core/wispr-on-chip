"""Model descriptions and parameter/MAC accounting.

A model is described as a list of pipeline *stages*. Each stage owns a set of
hardwired weight matrices and processes *items* (audio frames for encoder-side
stages, text tokens for decoder-side stages). The mapper places each stage on
its own block of tiles, so the accounting here is deliberately per stage.

Whisper presets reproduce the Hugging Face checkpoint parameter counts to
within ~0.5 % (biases and LayerNorm gains are ignored; negligible for area).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """One pipeline stage: a block of hardwired weights and its working state."""

    name: str
    kind: str
    """One of ``stem``, ``encoder``, ``decoder``, ``head``, ``embed``."""
    params: int
    """Hardwired weights in this stage."""
    linear_macs_per_item: int
    """MACs through the hardwired weights per frame/token."""
    attn_macs_per_item: int
    """MACs against the KV cache per frame/token (worst-case context)."""
    window_macs: int
    """MACs through hardwired weights done once per window (e.g. cross-attention K/V projection)."""
    kv_elements_per_stream: int
    """KV-cache elements a stage holds per in-flight stream (multiply by activation bits)."""
    act_elements_per_item: int
    """Working activation elements per item (residual stream, FFN hidden, etc.)."""

    @property
    def encoder_side(self) -> bool:
        return self.kind in ("stem", "encoder")

    @property
    def decoder_side(self) -> bool:
        return self.kind in ("decoder", "head", "embed")


@dataclass(frozen=True)
class ModelSpec:
    """Architecture hyper-parameters of an encoder-decoder or decoder-only transformer."""

    name: str
    family: str
    """``whisper`` (audio encoder + text decoder) or ``llm`` (decoder only)."""
    d_model: int
    n_heads: int
    n_enc_layers: int
    n_dec_layers: int
    vocab: int
    ffn_dim: int
    n_kv_heads: int | None = None
    gated_ffn: bool = False
    """SwiGLU-style FFN with three matrices instead of two."""
    tied_embeddings: bool = True
    n_mels: int = 0
    n_audio_ctx: int = 0
    """Encoder frames per window (after the conv stem)."""
    n_text_ctx: int = 448
    """Maximum decoder context (tokens) the KV cache must hold."""
    window_seconds: float = 0.0
    """Audio seconds per encoder window (Whisper: 30 s). 0 for decoder-only models."""
    tokens_per_window: int = 0
    """Typical decoded tokens per window; used for throughput in audio terms."""
    reference_params: int | None = None
    """Published parameter count, for sanity checks and reporting."""
    notes: str = ""

    # ---- derived dimensions -------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def d_kv(self) -> int:
        """Width of the K (or V) projection output."""
        return self.kv_heads * self.head_dim

    @property
    def has_encoder(self) -> bool:
        return self.n_enc_layers > 0

    # ---- per-block parameter counts ----------------------------------------
    def _attn_params(self, cross: bool = False) -> int:
        d, dkv = self.d_model, self.d_kv
        return 2 * d * d + 2 * d * dkv  # q, o are d x d; k, v are d x d_kv

    def _ffn_params(self) -> int:
        n = 3 if self.gated_ffn else 2
        return n * self.d_model * self.ffn_dim

    def encoder_layer_params(self) -> int:
        return self._attn_params() + self._ffn_params()

    def decoder_layer_params(self) -> int:
        cross = self._attn_params(cross=True) if self.has_encoder else 0
        return self._attn_params() + cross + self._ffn_params()

    def stem_params(self) -> int:
        if not self.has_encoder:
            return 0
        d = self.d_model
        return 3 * self.n_mels * d + 3 * d * d  # conv1 (k=3) + conv2 (k=3, stride 2)

    def embedding_params(self) -> int:
        return self.vocab * self.d_model

    def positional_params(self) -> int:
        """Learned/sinusoidal position tables (audio + text), stored as lookup ROM."""
        return (self.n_audio_ctx + self.n_text_ctx) * self.d_model

    # ---- stages -------------------------------------------------------------
    def stages(self) -> list[Stage]:
        """Pipeline stages in dataflow order: stem, encoder..., embed, decoder..., head."""
        d, dkv = self.d_model, self.d_kv
        out: list[Stage] = []

        if self.has_encoder:
            stem = self.stem_params()
            out.append(
                Stage(
                    name="stem",
                    kind="stem",
                    params=stem + self.n_audio_ctx * d,  # conv weights + audio position table
                    # conv1 runs on 2x the frames (before the stride-2 conv2)
                    linear_macs_per_item=2 * 3 * self.n_mels * d + 3 * d * d,
                    attn_macs_per_item=0,
                    window_macs=0,
                    kv_elements_per_stream=0,
                    act_elements_per_item=4 * d,
                )
            )
            for i in range(self.n_enc_layers):
                out.append(
                    Stage(
                        name=f"enc{i}",
                        kind="encoder",
                        params=self.encoder_layer_params(),
                        linear_macs_per_item=self.encoder_layer_params(),
                        attn_macs_per_item=2 * self.n_audio_ctx * d,
                        window_macs=0,
                        kv_elements_per_stream=2 * self.n_audio_ctx * dkv,
                        act_elements_per_item=2 * d + self.ffn_dim,
                    )
                )

        if not self.tied_embeddings:
            out.append(
                Stage(
                    name="embed",
                    kind="embed",
                    params=self.embedding_params(),
                    linear_macs_per_item=0,
                    attn_macs_per_item=0,
                    window_macs=0,
                    kv_elements_per_stream=0,
                    act_elements_per_item=d,
                )
            )

        for i in range(self.n_dec_layers):
            self_kv = 2 * self.n_text_ctx * dkv
            cross_kv = 2 * self.n_audio_ctx * dkv if self.has_encoder else 0
            cross_proj = 2 * self.n_audio_ctx * d * dkv if self.has_encoder else 0
            out.append(
                Stage(
                    name=f"dec{i}",
                    kind="decoder",
                    params=self.decoder_layer_params(),
                    linear_macs_per_item=self.decoder_layer_params() - 2 * d * dkv * (1 if self.has_encoder else 0),
                    attn_macs_per_item=2 * self.n_text_ctx * d + (2 * self.n_audio_ctx * d if self.has_encoder else 0),
                    window_macs=cross_proj,
                    kv_elements_per_stream=self_kv + cross_kv,
                    act_elements_per_item=2 * d + self.ffn_dim * (2 if self.gated_ffn else 1),
                )
            )

        out.append(
            Stage(
                name="head",
                kind="head",
                params=self.embedding_params() + self.n_text_ctx * d,  # (tied) embedding + text position table
                linear_macs_per_item=self.embedding_params(),
                attn_macs_per_item=0,
                window_macs=0,
                kv_elements_per_stream=0,
                act_elements_per_item=self.vocab,
            )
        )
        return out

    @property
    def total_params(self) -> int:
        return sum(s.params for s in self.stages())

    @property
    def frames_per_window(self) -> int:
        return self.n_audio_ctx

    def items_per_window(self, stage: Stage) -> int:
        """How many items a stage processes per window (frames or tokens)."""
        if stage.encoder_side:
            return self.n_audio_ctx
        return self.tokens_per_window

    def describe(self) -> str:
        lines = [f"{self.name} ({self.family})"]
        lines.append(
            f"  d_model={self.d_model} heads={self.n_heads} kv_heads={self.kv_heads} ffn={self.ffn_dim}"
            f" enc_layers={self.n_enc_layers} dec_layers={self.n_dec_layers} vocab={self.vocab}"
        )
        if self.has_encoder:
            lines.append(
                f"  audio: {self.n_mels} mels, {self.n_audio_ctx} frames / {self.window_seconds:g} s window,"
                f" ~{self.tokens_per_window} tokens per window"
            )
        lines.append(f"  text context: {self.n_text_ctx} tokens")
        ref = f" (published {self.reference_params/1e6:,.0f} M)" if self.reference_params else ""
        lines.append(f"  params: {self.total_params/1e6:,.1f} M{ref}")
        if self.notes:
            lines.append(f"  {self.notes}")
        return "\n".join(lines)


def _whisper(name: str, d: int, heads: int, layers: int, ref: int) -> ModelSpec:
    return ModelSpec(
        name=name,
        family="whisper",
        d_model=d,
        n_heads=heads,
        n_enc_layers=layers,
        n_dec_layers=layers,
        vocab=51866,
        ffn_dim=4 * d,
        n_mels=128 if name.endswith("v3") or name.endswith("turbo") else 80,
        n_audio_ctx=1500,
        n_text_ctx=448,
        window_seconds=30.0,
        tokens_per_window=90,
        reference_params=ref,
    )


MODELS: dict[str, ModelSpec] = {
    "whisper-tiny": _whisper("whisper-tiny", 384, 6, 4, 37_760_640),
    "whisper-base": _whisper("whisper-base", 512, 8, 6, 72_593_920),
    "whisper-small": _whisper("whisper-small", 768, 12, 12, 241_734_912),
    "whisper-medium": _whisper("whisper-medium", 1024, 16, 24, 763_857_920),
    "whisper-large-v3": _whisper("whisper-large-v3", 1280, 20, 32, 1_543_304_960),
    # large-v3-turbo keeps the 32-layer encoder and prunes the decoder to 4 layers.
    "whisper-large-v3-turbo": ModelSpec(
        name="whisper-large-v3-turbo",
        family="whisper",
        d_model=1280,
        n_heads=20,
        n_enc_layers=32,
        n_dec_layers=4,
        vocab=51866,
        ffn_dim=5120,
        n_mels=128,
        n_audio_ctx=1500,
        n_text_ctx=448,
        window_seconds=30.0,
        tokens_per_window=90,
        reference_params=808_650_240,
    ),
    # Decoder-only reference point, roughly the model Taalas showed on its first hardwired chip.
    "llama-3.1-8b": ModelSpec(
        name="llama-3.1-8b",
        family="llm",
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        n_enc_layers=0,
        n_dec_layers=32,
        vocab=128256,
        ffn_dim=14336,
        gated_ffn=True,
        tied_embeddings=False,
        n_text_ctx=4096,
        reference_params=8_030_000_000,
        notes="Context capped at 4096 tokens for KV-cache sizing; raise n_text_ctx to model longer prompts.",
    ),
}


def get_model(name: str) -> ModelSpec:
    key = name.lower()
    if key in MODELS:
        return MODELS[key]
    if f"whisper-{key}" in MODELS:
        return MODELS[f"whisper-{key}"]
    raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
