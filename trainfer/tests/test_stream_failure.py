"""A failed generation thread must release both the stream and model lock."""

import threading

import pytest
import torch

from trainfer.engine.inference import generate_chat_stream
from trainfer.scheduling import ModelScheduler

pytestmark = pytest.mark.cpu_only


def test_generate_failure_terminates_stream_and_releases_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            raise RuntimeError("synthetic generation failure")

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, *args, **kwargs):
            return "hello"

        def __call__(self, **kwargs):
            return {"input_ids": torch.tensor([[2, 3]])}

    lock = ModelScheduler()
    errors = []

    def consume():
        try:
            list(generate_chat_stream(Model(), Tokenizer(), [], mode_lock=lock))
        except RuntimeError as exc:
            errors.append(str(exc))

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(timeout=5)
    assert not consumer.is_alive(), "failed generation left the stream blocked"
    assert errors == ["synthetic generation failure"]
    assert lock.acquire(timeout=0.1)
    lock.release()
