"""
SLURPDataset: a lightweight, dependency-free dataset wrapper around the raw
SLURP jsonl annotation files.

This intentionally does NOT subclass torch.utils.data.Dataset — it's kept
as plain Python so it can be used in notebooks, profiling scripts, and the
data-audit stage without requiring torch to be installed. A thin torch
wrapper (for actual training) can subclass or wrap this later in
src/asr/dataset.py or src/intent/dataset.py.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional, TypedDict, Union

PathLike = Union[str, Path]

_SPLIT_FILES = {
    "train": "train.jsonl",
    "train_synthetic": "train_synthetic.jsonl",
    "validation": "devel.jsonl",
    "devel": "devel.jsonl",
    "dev": "devel.jsonl",
    "test": "test.jsonl",
}


class Sample(TypedDict):
    id: Optional[str]
    audio_path: Optional[Path]
    audio_exists: bool
    transcript: Optional[str]
    scenario: Optional[str]
    action: Optional[str]
    intent: Optional[str]
    split: str


class SLURPDataset:
    """Indexable, iterable view over one SLURP split.

    Each sample is a dict:
        {
            "id": str,
            "audio_path": Path,
            "audio_exists": bool,
            "transcript": str,
            "scenario": str,
            "action": str,
            "intent": str,
            "split": str,
        }

    Args:
        split: one of "train", "validation" (or "devel"/"dev"), "test".
            Case- and whitespace-insensitive.
        root_dir: path to the SLURP root (contains `dataset/` and `audio/`).
        include_synthetic: if split == "train" and this is True, also loads
            train_synthetic.jsonl and appends it (audio resolved against
            audio/slurp_synth). Default False, matching "real speech only"
            as your safest baseline training set. There's no equivalent flag
            for validation/test since SLURP only provides synthetic audio
            for train.
    """

    def __init__(
        self,
        split: str = "train",
        root_dir: PathLike = "data/raw/slurp",
        include_synthetic: bool = False,
    ):
        split = split.lower().strip()
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of: "
                f"{sorted(set(_SPLIT_FILES) - {'train_synthetic'})}"
            )

        self.split = split
        self.root_dir = Path(root_dir)
        self.annot_dir = self.root_dir / "dataset" / "slurp"
        self.audio_real_dir = self.root_dir / "audio" / "slurp_real"
        self.audio_synth_dir = self.root_dir / "audio" / "slurp_synth"

        self.samples: List[dict] = []
        self._load(split)

        if split == "train" and include_synthetic:
            self._load("train_synthetic", label_as="train")

    def _load(self, split_key: str, label_as: Optional[str] = None) -> None:
        jsonl_path = self.annot_dir / _SPLIT_FILES[split_key]
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {jsonl_path}\n"
                f"Check that root_dir='{self.root_dir}' points at the SLURP "
                f"folder containing dataset/slurp/*.jsonl"
            )

        audio_dir = self.audio_synth_dir if split_key == "train_synthetic" else self.audio_real_dir
        split_label = label_as or split_key

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._add_record(rec, audio_dir, split_label)

    def _add_record(self, rec: dict, audio_dir: Path, split_label: str) -> None:
        slurp_id = rec.get("slurp_id")
        sample_id = str(slurp_id) if slurp_id is not None else None
        transcript = rec.get("sentence")
        scenario = rec.get("scenario")
        action = rec.get("action")
        intent = rec.get("intent") or (
            f"{scenario}_{action}" if scenario and action else None
        )
        recordings = rec.get("recordings", [])

        if not recordings:
            self.samples.append({
                "id": sample_id,
                "audio_path": None,
                "audio_exists": False,
                "transcript": transcript,
                "scenario": scenario,
                "action": action,
                "intent": intent,
                "split": split_label,
            })
            return

        for r in recordings:
            fname = r.get("file")
            audio_path = (audio_dir / fname) if fname else None
            self.samples.append({
                "id": sample_id,
                "audio_path": audio_path,
                "audio_exists": bool(audio_path and audio_path.exists()),
                "transcript": transcript,
                "scenario": scenario,
                "action": action,
                "intent": intent,
                "split": split_label,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.samples)

    def __repr__(self) -> str:
        return f"SLURPDataset(split='{self.split}', n_samples={len(self)})"

    def summary(self) -> dict:
        """Small dict of quick stats: sample count, unique intents, and how
        many samples are missing an audio_path, transcript, or intent.
        Meant for a one-line sanity check when loading a split, not a full
        audit (see notebooks/01_dataset_audit.ipynb for that).

        Note: n_missing_audio counts samples with no audio_path at all
        (malformed record). To find paths that are set but point to files
        that don't actually exist on disk, check audio_exists per-sample
        instead, e.g. sum(not s["audio_exists"] for s in dataset).
        """
        n_missing_audio = sum(1 for s in self.samples if not s["audio_path"])
        n_missing_transcript = sum(1 for s in self.samples if not s["transcript"])
        n_missing_intent = sum(1 for s in self.samples if not s["intent"])
        intents = {s["intent"] for s in self.samples if s["intent"]}

        return {
            "split": self.split,
            "n_samples": len(self.samples),
            "n_unique_intents": len(intents),
            "n_missing_audio": n_missing_audio,
            "n_missing_transcript": n_missing_transcript,
            "n_missing_intent": n_missing_intent,
        }

    def intent_distribution(self) -> Counter:
        """Count of samples per intent, as a Counter (most_common() etc. work)."""
        return Counter(s["intent"] for s in self.samples if s["intent"])
