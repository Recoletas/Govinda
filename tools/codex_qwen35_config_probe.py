# AI-generated, awaiting verification by recoletas on 2026-07-07.
import json

from transformers import AutoConfig

MODEL = "/public/home/xdzs2026_c087/Qwen3.5-27B"


def dump_obj(prefix: str, obj) -> None:
    for key in [
        "model_type",
        "architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_experts",
        "layer_types",
        "vocab_size",
        "tie_word_embeddings",
        "key_dim",
        "value_dim",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ]:
        print(prefix, key, getattr(obj, key, None))


def main() -> None:
    raw = json.load(open(f"{MODEL}/config.json"))
    for key in [
        "model_type",
        "architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_experts",
        "layer_types",
        "vocab_size",
        "tie_word_embeddings",
    ]:
        print("json", key, raw.get(key))

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    dump_obj("auto", cfg)
    text = getattr(cfg, "text_config", cfg)
    dump_obj("text", text)


if __name__ == "__main__":
    main()
