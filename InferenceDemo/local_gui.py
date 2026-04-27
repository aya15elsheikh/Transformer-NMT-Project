from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

import torch
import torch.nn as nn


PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class TranslationTokenizer:
    def __init__(
        self,
        max_len: int,
        ar_stoi: Dict[str, int],
        ar_itos: List[str],
        en_stoi: Dict[str, int],
        en_itos: List[str],
    ) -> None:
        self.max_len = max_len
        self.ar_stoi = ar_stoi
        self.ar_itos = ar_itos
        self.en_stoi = en_stoi
        self.en_itos = en_itos
        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    @classmethod
    def load(cls, tokenizer_dir: Path) -> "TranslationTokenizer":
        tokenizer_path = tokenizer_dir / "tokenizer.pkl"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at: {tokenizer_path}\n"
                "Run notebook pipeline and save tokenizer.pkl first."
            )

        with tokenizer_path.open("rb") as file_handle:
            state = pickle.load(file_handle)

        return cls(
            max_len=int(state["max_len"]),
            ar_stoi=state["ar_stoi"],
            ar_itos=state["ar_itos"],
            en_stoi=state["en_stoi"],
            en_itos=state["en_itos"],
        )

    def encode_sentence(self, sentence: str, lang: str) -> torch.Tensor:
        stoi = self.ar_stoi if lang == "ar" else self.en_stoi
        unk_id = stoi[UNK_TOKEN]
        tokens = str(sentence).split()
        indices = [stoi.get(token, unk_id) for token in tokens]

        sequence = [stoi[SOS_TOKEN]] + indices + [stoi[EOS_TOKEN]]
        if len(sequence) > self.max_len:
            sequence = sequence[: self.max_len - 1] + [stoi[EOS_TOKEN]]

        if len(sequence) < self.max_len:
            sequence = sequence + [stoi[PAD_TOKEN]] * (self.max_len - len(sequence))

        return torch.tensor(sequence, dtype=torch.long)

    def decode(self, indices: List[int], lang: str, skip_special: bool = True) -> str:
        itos = self.ar_itos if lang == "ar" else self.en_itos
        output_tokens: List[str] = []

        for idx in indices:
            if idx < 0 or idx >= len(itos):
                continue
            token = itos[idx]
            if skip_special and token in self.special_tokens:
                continue
            output_tokens.append(token)

        return " ".join(output_tokens)


class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        out, attn = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc(out), attn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn1, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        attn2, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, max_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.dropout(self.pos(x))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, max_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.dropout(self.pos(x))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        max_len: int = 50,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, n_layers, n_heads, d_ff, max_len, dropout)

    def make_src_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        t_len = tgt.size(1)
        return torch.tril(torch.ones(t_len, t_len, device=tgt.device)).bool()

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        src_mask = self.make_src_mask(src, pad_idx)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def greedy_decode(
    model: Transformer,
    src_tensor: torch.Tensor,
    tokenizer: TranslationTokenizer,
    device: torch.device,
    max_len: int = 50,
) -> str:
    model.eval()
    sos_id = tokenizer.ar_stoi[SOS_TOKEN]
    eos_id = tokenizer.ar_stoi[EOS_TOKEN]
    src_pad_id = tokenizer.en_stoi[PAD_TOKEN]

    src = src_tensor.to(device)
    dec_input = torch.tensor([[sos_id]], device=device)

    with torch.no_grad():
        src_mask = model.make_src_mask(src, src_pad_id)
        enc_out = model.encoder(src, src_mask)

        for _ in range(max_len):
            tgt_mask = model.make_tgt_mask(dec_input)
            out = model.decoder(dec_input, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
            next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
            dec_input = torch.cat([dec_input, next_tok], dim=1)
            if next_tok.item() == eos_id:
                break

    return tokenizer.decode(dec_input[0].tolist()[1:], "ar", skip_special=True)


def beam_search_decode(
    model: Transformer,
    src_tensor: torch.Tensor,
    tokenizer: TranslationTokenizer,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 50,
    length_penalty: float = 0.7,
    no_repeat_ngram: int = 3,
) -> str:
    model.eval()
    sos_id = tokenizer.ar_stoi[SOS_TOKEN]
    eos_id = tokenizer.ar_stoi[EOS_TOKEN]
    src_pad_id = tokenizer.en_stoi[PAD_TOKEN]

    src = src_tensor.to(device)

    with torch.no_grad():
        src_mask = model.make_src_mask(src, src_pad_id)
        enc_out = model.encoder(src, src_mask)

        beams: List[Tuple[float, List[int]]] = [(0.0, [sos_id])]
        completed: List[Tuple[float, List[int]]] = []

        for _ in range(max_len):
            if not beams:
                break

            candidates: List[Tuple[float, List[int]]] = []

            for score, seq in beams:
                dec_input = torch.tensor([seq], device=device)
                tgt_mask = model.make_tgt_mask(dec_input)
                out = model.decoder(dec_input, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
                log_probs = torch.log_softmax(out[:, -1, :], dim=-1)[0]

                if no_repeat_ngram > 0 and len(seq) >= no_repeat_ngram:
                    prefix = tuple(seq[-(no_repeat_ngram - 1) :])
                    for idx in range(len(seq) - no_repeat_ngram + 1):
                        if tuple(seq[idx : idx + no_repeat_ngram - 1]) == prefix:
                            blocked_token = seq[idx + no_repeat_ngram - 1]
                            log_probs[blocked_token] = float("-inf")

                top_probs, top_ids = log_probs.topk(beam_size)
                for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
                    new_seq = seq + [token_id]
                    new_score = score + prob
                    if token_id == eos_id:
                        lp = ((5 + len(new_seq)) / 6) ** length_penalty
                        completed.append((new_score / lp, new_seq))
                    else:
                        candidates.append((new_score, new_seq))

            candidates.sort(key=lambda item: item[0], reverse=True)
            beams = candidates[:beam_size]

        if not completed:
            beams.sort(key=lambda item: item[0], reverse=True)
            best_seq = beams[0][1] if beams else [sos_id, eos_id]
            best_score = beams[0][0] if beams else 0.0
            completed.append((best_score, best_seq + [eos_id]))

        completed.sort(key=lambda item: item[0], reverse=True)
        best_seq = completed[0][1][1:]

    return tokenizer.decode(best_seq, "ar", skip_special=True)


class DesktopTranslatorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("English to Arabic Translator")
        self.root.geometry("900x650")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: TranslationTokenizer | None = None
        self.model: Transformer | None = None

        self._build_layout()
        self._load_runtime_or_fail()

    def _build_layout(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="English Input", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.input_text = tk.Text(frame, height=8, wrap=tk.WORD, font=("Segoe UI", 11))
        self.input_text.pack(fill=tk.X, pady=(6, 12))

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, pady=(0, 10))

        self.decode_method = tk.StringVar(value="greedy")
        ttk.Radiobutton(controls, text="Greedy", variable=self.decode_method, value="greedy").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(controls, text="Beam", variable=self.decode_method, value="beam").pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(controls, text="Max Len:").pack(side=tk.LEFT)
        self.max_len_entry = ttk.Entry(controls, width=6)
        self.max_len_entry.insert(0, "50")
        self.max_len_entry.pack(side=tk.LEFT, padx=(6, 16))

        ttk.Label(controls, text="Beam Size:").pack(side=tk.LEFT)
        self.beam_size_entry = ttk.Entry(controls, width=6)
        self.beam_size_entry.insert(0, "4")
        self.beam_size_entry.pack(side=tk.LEFT, padx=(6, 16))

        ttk.Button(controls, text="Translate", command=self._on_translate).pack(side=tk.RIGHT)

        ttk.Label(frame, text="Arabic Output", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.output_text = tk.Text(frame, height=10, wrap=tk.WORD, font=("Segoe UI", 13), state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(6, 10))

        self.status_var = tk.StringVar(value="Loading model...")
        ttk.Label(frame, textvariable=self.status_var).pack(anchor="w")

    def _load_runtime_or_fail(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        tokenizer_dir = repo_root / "models" / "tokenizer"
        checkpoint_path = repo_root / "models" / "checkpoints" / "best_model.pt"

        try:
            self.tokenizer = TranslationTokenizer.load(tokenizer_dir)
            self.model = Transformer(
                src_vocab=len(self.tokenizer.en_stoi),
                tgt_vocab=len(self.tokenizer.ar_stoi),
                d_model=256,
                n_heads=8,
                n_layers=3,
                d_ff=512,
                max_len=50,
                dropout=0.1,
            ).to(self.device)

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

            raw_checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = raw_checkpoint.get("model_state_dict", raw_checkpoint) if isinstance(raw_checkpoint, dict) else raw_checkpoint
            state_dict = _strip_module_prefix(state_dict)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()

            self.status_var.set(f"Ready. Device: {self.device}. Checkpoint: {checkpoint_path.name}")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            self.root.destroy()

    def _on_translate(self) -> None:
        if self.model is None or self.tokenizer is None:
            messagebox.showerror("Error", "Model is not loaded.")
            return

        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input required", "Please enter English text.")
            return

        try:
            max_len = max(5, int(self.max_len_entry.get().strip()))
            beam_size = max(2, int(self.beam_size_entry.get().strip()))
        except ValueError:
            messagebox.showwarning("Invalid parameters", "Max Len and Beam Size must be integers.")
            return

        src_tensor = self.tokenizer.encode_sentence(text, "en").unsqueeze(0)

        if self.decode_method.get() == "beam":
            arabic_text = beam_search_decode(
                model=self.model,
                src_tensor=src_tensor,
                tokenizer=self.tokenizer,
                device=self.device,
                beam_size=beam_size,
                max_len=max_len,
                length_penalty=0.7,
                no_repeat_ngram=3,
            )
        else:
            arabic_text = greedy_decode(
                model=self.model,
                src_tensor=src_tensor,
                tokenizer=self.tokenizer,
                device=self.device,
                max_len=max_len,
            )

        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", arabic_text)
        self.output_text.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    DesktopTranslatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
