#!/usr/bin/env python3
"""Convert the Turp HF checkpoint to GGUF using llama.cpp's converter.

Turp trains its own byte-level BPE vocabulary, so llama.cpp cannot fingerprint it
against its list of known tokenizers. The pre-tokenization itself is exactly the
GPT-2 byte-level scheme, so we register it as "gpt2" when the hash lookup fails.
"""
import runpy
import sys

LLAMA = "/tmp/llama.cpp"


def main():
    sys.path.insert(0, LLAMA)
    sys.path.insert(0, LLAMA + "/gguf-py")

    import conversion
    import conversion.base as base

    patched_any = False
    for module in (base, conversion):
        for name in dir(module):
            cls = getattr(module, name)
            if not isinstance(cls, type):
                continue
            if "get_vocab_base_pre" not in getattr(cls, "__dict__", {}):
                continue
            original = cls.get_vocab_base_pre

            def patched(self, tokenizer, _original=original):
                try:
                    return _original(self, tokenizer)
                except NotImplementedError:
                    print("turp: unknown BPE fingerprint, using gpt2 byte-level pre-tokenizer")
                    return "gpt2"

            cls.get_vocab_base_pre = patched
            print(f"turp: patched {cls.__module__}.{cls.__name__}.get_vocab_base_pre")
            patched_any = True

    if not patched_any:
        raise SystemExit("could not find get_vocab_base_pre to patch")

    sys.argv = [
        "convert_hf_to_gguf.py",
        "work/model",
        "--outfile",
        "turp-v1.gguf",
        "--outtype",
        "f16",
    ]
    runpy.run_path(LLAMA + "/convert_hf_to_gguf.py", run_name="__main__")


if __name__ == "__main__":
    main()
