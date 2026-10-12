"""A tiny randomly initialized chat model built on the spot, so training tests need no download."""

from pathlib import Path

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CORPUS = [
    "user assistant system The Witcher 3 review: a great open world game.",
    "Hint: use Igni against drowners. Boss fight walkthrough and trailer.",
    "Відьмак 3 огляд: чудова гра з відкритим світом. Підказка: використовуйте знак Ігні.",
    "Ведьмак 3 обзор: отличная игра. Подсказка: используйте знак Игни.",
]


def build_tiny_model(path: Path) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(CORPUS * 10, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        chat_template=CHATML,
    )
    fast.save_pretrained(path)
    config = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        # Large enough weights that the frozen output layer can tell tokens apart, so a LoRA
        # adapter visibly learns.
        initializer_range=0.5,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    Qwen2ForCausalLM(config).save_pretrained(path)
    return path
