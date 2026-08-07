"""
ASR model loading and single-file inference.

Promoted from notebooks/03_baseline_asr.ipynb and notebooks/04_asr_evaluation.ipynb,
where the model-loading cell was byte-identical and transcribe() was functionally
identical (only a docstring/sanity-check line differed) across both notebooks —
the signal, per the project's research-to-refactor workflow, that this code has
stopped changing and is safe to centralize.

Kept deliberately generic (model/processor/device are parameters, not module-level
globals) so this same code works for both the baseline pretrained model (03/04)
and the fine-tuned model (05/06) without any changes.
"""

from typing import Optional

import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

from src.utils.audio import load_audio


def load_asr_model(
    model_name: str = "facebook/wav2vec2-base-960h",
    device: Optional[str] = None,
):
    """Load a Wav2Vec2 CTC model + processor and move the model to `device`.

    Args:
        model_name: HuggingFace model id or local path. Pass the same
            pretrained id for baseline runs, or your fine-tuned checkpoint
            directory (e.g. "models/wav2vec2-slurp-finetuned") for 05/06.
        device: "cuda" or "cpu". Auto-detects if not given.

    Returns:
        (processor, model, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)

    processor = AutoProcessor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return processor, model, device


def transcribe(audio_path, model, processor, device: str = "cpu") -> Optional[str]:
    """Run greedy CTC inference on one audio file.

    Args:
        audio_path: path to the audio file (any format soundfile can read).
        model: a loaded Wav2Vec2ForCTC (from load_asr_model).
        processor: the matching AutoProcessor (from load_asr_model).
        device: device the model lives on.

    Returns:
        Greedy-decoded transcript as a string, or None if the audio couldn't
        be loaded (missing/corrupted file) — callers can skip these rather
        than crash a batch run.
    """
    try:
        waveform, sr = load_audio(audio_path, target_sr=16_000)
    except Exception as e:
        print(f"Failed to load {audio_path}: {e}")
        return None

    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    prediction = processor.batch_decode(predicted_ids)[0]
    return prediction
