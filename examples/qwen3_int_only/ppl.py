import argparse
from html.parser import HTMLParser
import math
from pathlib import Path
from urllib.request import urlopen

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.qwen3_int_only.model import Qwen3FloatModel, Qwen3IntOnlyModel


DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog. Language models estimate the probability of each next token.",
    "Qwen3 is a dense transformer model. This example measures perplexity on a small deterministic text batch.",
    "Integer-only inference should be judged by end-to-end perplexity against the floating point baseline.",
]
PROTO_STAGES = ["float", "q15-io", "rms", "qkv", "qknorm-rope", "attn-q15", "attn-i12-out", "o-proj", "post-rms", "gate-up", "silu-mul", "down"]


DATA_DIR = Path(__file__).resolve().parent / "data"


class ParagraphParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_p = False
        self.cur = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.in_p = True
            self.cur = []

    def handle_endtag(self, tag):
        if tag == "p" and self.in_p:
            text = " ".join("".join(self.cur).split())
            if text:
                self.paragraphs.append(text)
            self.in_p = False

    def handle_data(self, data):
        if self.in_p:
            self.cur.append(data)


def archives_transcript(url, start_prefix, end_prefix):
    html = urlopen(url, timeout=20).read().decode()
    parser = ParagraphParser()
    parser.feed(html)
    start = next(i for i, p in enumerate(parser.paragraphs) if p.startswith(start_prefix))
    end = next(i for i, p in enumerate(parser.paragraphs[start:], start) if p.startswith(end_prefix))
    return "\n\n".join(parser.paragraphs[start:end])


def archives_declaration_text():
    path = DATA_DIR / "declaration_of_independence.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return archives_transcript(
        "https://www.archives.gov/founding-docs/declaration-transcript",
        "In Congress, July 4, 1776",
        "Back to Main",
    )


def archives_founding_text():
    path = DATA_DIR / "founding_docs.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    constitution = archives_transcript(
        "https://www.archives.gov/founding-docs/constitution-transcript",
        "We the People",
        "Back to Main",
    )
    return archives_declaration_text() + "\n\n" + constitution


def read_texts(paths):
    return [open(path, encoding="utf-8").read() for path in paths]


def tokens_from_texts(tokenizer, texts, max_tokens, device, repeat_text=False):
    ids = []
    while len(ids) < max_tokens:
        for text in texts:
            ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
        if not repeat_text:
            break
    ids = ids[:max_tokens]
    return torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def hf_ppl(model_dir, texts, max_tokens, repeat_text):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to("cuda")
    input_ids = tokens_from_texts(tokenizer, texts, max_tokens, model.device, repeat_text=repeat_text)
    logits = model(input_ids[:, :-1]).logits.float()
    labels = input_ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return math.exp(float(loss)), float(loss), int(labels.numel())


@torch.no_grad()
def int_only_ppl(model_dir, texts, max_tokens, layers, verbose, hybrid, repeat_text):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    input_ids = tokens_from_texts(tokenizer, texts, max_tokens, "cuda", repeat_text=repeat_text).squeeze(0)
    model = Qwen3IntOnlyModel(input_ids.numel() - 1, model_dir=model_dir)
    logits = model.logits(input_ids[:-1], layers=layers, verbose=verbose, hybrid=hybrid or "int12-attn-q15").float()
    labels = input_ids[1:]
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return math.exp(float(loss)), float(loss), int(labels.numel())


@torch.no_grad()
def local_float_ppl(model_dir, texts, max_tokens, layers, verbose, repeat_text):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    input_ids = tokens_from_texts(tokenizer, texts, max_tokens, "cuda", repeat_text=repeat_text).squeeze(0)
    model = Qwen3FloatModel(input_ids.numel() - 1, model_dir=model_dir)
    logits = model.logits(input_ids[:-1], layers=layers, verbose=verbose).float()
    labels = input_ids[1:]
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return math.exp(float(loss)), float(loss), int(labels.numel())


@torch.no_grad()
def proto_stage_ppl(model_dir, texts, max_tokens, stage, repeat_text):
    from examples.qwen3_int_only.diagnose_staged import run_stage

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    input_ids = tokens_from_texts(tokenizer, texts, max_tokens, "cuda", repeat_text=repeat_text).squeeze(0)
    labels = input_ids[1:]
    model = Qwen3FloatModel(input_ids.numel() - 1, model_dir=model_dir)
    _, loss = run_stage(model, input_ids[:-1], labels, stage)
    return math.exp(float(loss)), float(loss), int(labels.numel())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--backend", choices=["hf", "local-float", "proto-stage", "int-only"], default="local-float")
    parser.add_argument("--max-tokens", type=int, default=2049)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--hybrid")
    parser.add_argument("--stage", choices=PROTO_STAGES, default="float")
    parser.add_argument("--corpus", choices=["default", "declaration", "founding"], default="default")
    parser.add_argument("--repeat-text", action="store_true", default=None)
    parser.add_argument("--no-repeat-text", action="store_false", dest="repeat_text")
    parser.add_argument("--text-file", action="append")
    parser.add_argument("--text", action="append")
    args = parser.parse_args()
    if args.text_file:
        texts = read_texts(args.text_file)
    elif args.text:
        texts = args.text
    elif args.corpus == "declaration":
        texts = [archives_declaration_text()]
    elif args.corpus == "founding":
        texts = [archives_founding_text()]
    else:
        texts = DEFAULT_TEXTS
    repeat_text = args.repeat_text if args.repeat_text is not None else args.corpus == "default" and not args.text and not args.text_file
    if args.backend == "hf":
        ppl, loss, ntokens = hf_ppl(args.model_dir, texts, args.max_tokens, repeat_text)
    elif args.backend == "local-float":
        ppl, loss, ntokens = local_float_ppl(args.model_dir, texts, args.max_tokens, args.layers, args.verbose, repeat_text)
    elif args.backend == "proto-stage":
        ppl, loss, ntokens = proto_stage_ppl(args.model_dir, texts, args.max_tokens, args.stage, repeat_text)
    else:
        ppl, loss, ntokens = int_only_ppl(args.model_dir, texts, args.max_tokens, args.layers, args.verbose, args.hybrid, repeat_text)
    suffix = f" stage={args.stage}" if args.backend == "proto-stage" else f" hybrid={args.hybrid or 'int12-attn-q15'}" if args.backend == "int-only" else ""
    print(f"backend={args.backend}{suffix} tokens={ntokens} loss={loss:.6f} ppl={ppl:.6f}")


if __name__ == "__main__":
    main()
