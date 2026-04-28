from __future__ import annotations

from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

import torch

import model_runtime as runtime


class DesktopTranslatorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("English to Arabic Translator")
        self.root.geometry("900x650")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: runtime.TranslationTokenizer | None = None
        self.model: runtime.Transformer | None = None
        self.checkpoint_path: Path | None = None

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
        try:
            self.tokenizer, self.model, self.checkpoint_path = runtime.load_runtime(
                repo_root=repo_root,
                device=self.device,
            )
            self.max_len_entry.delete(0, tk.END)
            self.max_len_entry.insert(0, str(self.tokenizer.max_len))
            self.status_var.set(
                "Ready. "
                f"Device: {self.device}. "
                f"Checkpoint: {self.checkpoint_path.name}"
            )
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
            arabic_text = runtime.beam_search_decode(
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
            arabic_text = runtime.greedy_decode(
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
