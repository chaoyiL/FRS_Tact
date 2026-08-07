# Vendored from openpi (Apache-2.0), src/openpi/models/tokenizer.py,
# commit 15a9616a00943ada6c20a0f158e3adb39df2ccac (2026-06-16), TRIMMED.
#
# Only `PaligemmaTokenizer` is kept -- that's the one pi0.5 actually uses (pi0.py only ever calls
# `self.PaliGemma.llm(obs.tokenized_prompt, method="embed")`, never a FAST/binning/FSQ head).
# Upstream's `FASTTokenizer` / `BinningTokenizer` / `FSQTokenizer` classes are for pi0-FAST and
# RoboArena baselines, not pi0/pi0.5, and FSQTokenizer additionally needs
# `openpi.models.utils.fsq_tokenizer`, which was not vendored. See README.md in this directory.
#
# IMPORTANT for pi0.5 (`pi05=True`): unlike pi0, the state is *not* fed as a continuous input --
# it is discretized and baked into the tokenized prompt (`tokenize(prompt, state=state)` below).
# Whoever builds `Observation.tokenized_prompt`/`tokenized_prompt_mask` for pi0.5 must call this
# with the robot state, matching `Pi0Config.discrete_state_input` (defaults to `pi05`). See
# pi0.py:Pi0.embed_suffix (`if not self.pi05: state_token = self.state_proj(...)`).

import logging

import numpy as np
import sentencepiece

from . import download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)
