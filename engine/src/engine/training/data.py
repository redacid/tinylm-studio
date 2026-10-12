"""Turning chat examples into token ids for supervised fine-tuning.

The model learns only the assistant's answers: tokens of the system and user messages and of the
chat template around them get the label -100, which the loss ignores. The answers include the
template's end-of-turn marker, so the tuned model also learns where to stop.
"""

from dataclasses import dataclass

IGNORE = -100
PROBE = "␂probe␃"


def render(tokenizer, messages: list[dict], **options) -> str:
    # enable_thinking=False keeps Qwen3's empty reasoning block, the same as when chatting;
    # other templates ignore it.
    return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False, **options)


def end_of_turn(tokenizer) -> str:
    """The text the chat template puts after an assistant answer, such as <|im_end|>."""
    text = render(
        tokenizer, [{"role": "user", "content": "x"}, {"role": "assistant", "content": PROBE}]
    )
    return text[text.rfind(PROBE) + len(PROBE) :].strip()


def answer_spans(text: str, messages: list[dict], eot: str) -> list[tuple[int, int]]:
    """Character ranges of the assistant answers (with the end-of-turn marker) in the rendered
    conversation. Messages are found in order, so an answer repeated in a question is skipped."""
    spans, cursor = [], 0
    for message in messages:
        start = text.find(message["content"], cursor)
        if start < 0:
            # The template changed the text (e.g. stripped it); the rest cannot be located.
            break
        end = start + len(message["content"])
        if message["role"] == "assistant":
            if eot and text.startswith(eot, end):
                end += len(eot)
            spans.append((start, end))
        cursor = end
    return spans


@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]

    @property
    def trained_tokens(self) -> int:
        return sum(label != IGNORE for label in self.labels)


def encode(tokenizer, messages: list[dict], max_length: int, eot: str | None = None) -> Encoded:
    """Token ids and labels of one example, cut to max_length tokens."""
    if eot is None:
        eot = end_of_turn(tokenizer)
    text = render(tokenizer, messages)
    spans = answer_spans(text, messages, eot)
    tokens = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )
    ids, labels = [], []
    for token_id, (start, end) in zip(tokens["input_ids"], tokens["offset_mapping"], strict=True):
        ids.append(token_id)
        # Overlap, not containment: a token can join the answer's first word with the space or
        # newline before it.
        inside = any(start < e and end > s for s, e in spans)
        labels.append(token_id if inside else IGNORE)
    return Encoded(ids[:max_length], labels[:max_length])


def encode_all(tokenizer, examples: list[list[dict]], max_length: int) -> tuple[list[Encoded], int]:
    """Encodes examples, dropping those with nothing left to learn after cutting them to
    max_length. Returns the encoded examples and how many were dropped."""
    if not tokenizer.chat_template:
        raise ValueError("The base model has no chat template; pick an instruction-tuned model.")
    eot = end_of_turn(tokenizer)
    kept = [encode(tokenizer, messages, max_length, eot) for messages in examples]
    usable = [e for e in kept if e.trained_tokens > 0]
    return usable, len(kept) - len(usable)


class Collator:
    """Pads a batch to its longest example."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        longest = max(len(item["input_ids"]) for item in batch)
        ids, labels, mask = [], [], []
        for item in batch:
            pad = longest - len(item["input_ids"])
            ids.append(item["input_ids"] + [self.pad_id] * pad)
            labels.append(item["labels"] + [IGNORE] * pad)
            mask.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }
