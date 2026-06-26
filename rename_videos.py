#!/usr/bin/env python3
"""
rename_videos.py — renomeia vídeos automaticamente com data + slug gerado por IA.

Fluxo por arquivo:
  1. Extrai a data do nome do arquivo (suporte a DJI: DJI_YYYYMMDDHHMMSS_...)
  2. Transcreve os primeiros N segundos com faster-whisper (local, sem API)
  3. Manda o transcript para a IA escolhida e pede um slug descritivo
  4. Renomeia para DD-MM-YYYY-slug-descritivo.ext

Uso:
    python rename_videos.py /pasta/com/videos/
    python rename_videos.py /pasta/ --dry-run
    python rename_videos.py video.mov --provider ollama
    python rename_videos.py /pasta/ --yes   # sem confirmação

Provedores de IA (configurar em .env ou variáveis de ambiente):
    AI_PROVIDER=anthropic   → ANTHROPIC_API_KEY obrigatória (padrão)
    AI_PROVIDER=ollama      → local, sem API key; OLLAMA_MODEL=llama3
    AI_PROVIDER=cmd         → qualquer CLI; AI_CMD="gemini -p" (lê stdin)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import unicodedata
import urllib.request
import json
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".mts", ".m2ts", ".mxf", ".wmv"}

# DJI naming: DJI_YYYYMMDDHHMMSS_NNNN_X.ext
_DJI_RE = re.compile(r"DJI_(\d{4})(\d{2})(\d{2})\d+", re.IGNORECASE)

SLUG_PROMPT = """\
Você receberá a transcrição de um trecho de vídeo. Gere um slug descritivo \
em português brasileiro com 3 a 5 palavras, em kebab-case (sem acentos, \
tudo minúsculo, palavras separadas por hífen). Responda APENAS com o slug.

Exemplos válidos:
  falando-sobre-edicao-podcast
  dicas-para-editar-video
  review-camera-dji-osmo
  conversa-sobre-marketing-digital

Transcrição:
{transcript}"""


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Texto → kebab-case ASCII sem acentos."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text).strip("-")
    return text or "video"


def parse_date(path: Path) -> str:
    """Retorna DD-MM-YYYY extraído do nome do arquivo ou da data de modificação."""
    m = _DJI_RE.match(path.stem)
    if m:
        year, month, day = m.group(1), m.group(2), m.group(3)
        return f"{day}-{month}-{year}"
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return mtime.strftime("%d-%m-%Y")


# ---------------------------------------------------------------------------
# Transcrição (faster-whisper, local, sem API)
# ---------------------------------------------------------------------------

def transcribe(video: Path, model_name: str, max_seconds: float) -> str:
    """Transcreve até max_seconds do vídeo. Retorna o texto completo."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Erro: faster-whisper não instalado. Execute: pip install faster-whisper")

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(video), word_timestamps=False)

    parts: list[str] = []
    for seg in segments:
        parts.append(seg.text.strip())
        if seg.end >= max_seconds:
            break

    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Geração de slug via IA
# ---------------------------------------------------------------------------

def _slug_anthropic(transcript: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=30,
        messages=[{"role": "user", "content": SLUG_PROMPT.format(transcript=transcript[:800])}],
    )
    return slugify(resp.content[0].text.strip())


def _slug_ollama(transcript: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": SLUG_PROMPT.format(transcript=transcript[:800]),
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return slugify(json.loads(resp.read())["response"].strip())


def _slug_cmd(transcript: str, cmd: str) -> str:
    """Chama qualquer CLI que leia o prompt via stdin e escreva o slug em stdout."""
    result = subprocess.run(
        cmd, shell=True,
        input=SLUG_PROMPT.format(transcript=transcript[:800]),
        capture_output=True, text=True, timeout=60,
    )
    return slugify(result.stdout.strip())


def generate_slug(transcript: str, provider: str, ollama_model: str, ai_cmd: str) -> str:
    if not transcript:
        return "sem-transcricao"
    if provider == "anthropic":
        return _slug_anthropic(transcript)
    if provider == "ollama":
        return _slug_ollama(transcript, ollama_model)
    if provider == "cmd":
        if not ai_cmd:
            sys.exit("Erro: --ai-cmd obrigatório quando --provider=cmd")
        return _slug_cmd(transcript, ai_cmd)
    raise ValueError(f"Provider desconhecido: {provider}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Renomeia vídeos com data + slug gerado por IA via transcrição.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", help="Diretório ou arquivo de vídeo")
    ap.add_argument("--dry-run", action="store_true",
                    help="Mostra renomes propostos sem executar")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="Aplica renomes sem pedir confirmação")
    ap.add_argument("--provider", default=os.getenv("AI_PROVIDER", "anthropic"),
                    choices=["anthropic", "ollama", "cmd"],
                    help="Provedor de IA (padrão: anthropic)")
    ap.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3"),
                    help="Modelo ollama (padrão: llama3)")
    ap.add_argument("--ai-cmd", default=os.getenv("AI_CMD", ""),
                    help='Comando CLI para IA, ex: "gemini -p" (lê stdin)')
    ap.add_argument("--whisper-model", default=os.getenv("WHISPER_MODEL", "large-v3-turbo"),
                    help="Modelo faster-whisper (padrão: large-v3-turbo)")
    ap.add_argument("--max-seconds", type=float, default=90.0,
                    help="Segundos máximos de áudio a transcrever por arquivo (padrão: 90)")
    args = ap.parse_args()

    target = Path(args.path)
    if target.is_file():
        videos = [target] if target.suffix.lower() in VIDEO_EXTENSIONS else []
    elif target.is_dir():
        videos = sorted(f for f in target.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS)
    else:
        sys.exit(f"Caminho não encontrado: {target}")

    if not videos:
        print("Nenhum vídeo encontrado.")
        return

    print(f"{len(videos)} vídeo(s) encontrado(s) | IA: {args.provider} | "
          f"whisper: {args.whisper_model} | max: {args.max_seconds:.0f}s\n")

    renames: list[tuple[Path, Path]] = []

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")

        date_str = parse_date(video)
        print(f"  data   : {date_str}")

        print(f"  whisper: transcrevendo primeiros {args.max_seconds:.0f}s...")
        transcript = transcribe(video, args.whisper_model, args.max_seconds)
        preview = (transcript[:90] + "...") if len(transcript) > 90 else transcript
        print(f"  texto  : {preview or '(vazio)'}")

        print(f"  ia     : gerando slug ({args.provider})...")
        try:
            slug = generate_slug(transcript, args.provider,
                                 args.ollama_model, args.ai_cmd)
        except Exception as exc:
            print(f"  erro   : {exc}", file=sys.stderr)
            slug = "erro-slug"

        new_name = f"{date_str}-{slug}{video.suffix.lower()}"
        print(f"  novo   : {new_name}\n")
        renames.append((video, video.parent / new_name))

    # Resumo
    print("=" * 64)
    print("Renomes propostos:\n")
    for old, new in renames:
        print(f"  {old.name}")
        print(f"  → {new.name}\n")

    if args.dry_run:
        print("(--dry-run: nenhum arquivo foi alterado)")
        return

    if not args.yes:
        resp = input("Aplicar? [s/N] ").strip().lower()
        if resp not in ("s", "sim", "y", "yes"):
            print("Cancelado.")
            return

    print()
    for old, new in renames:
        if new.exists():
            print(f"  pulando: {new.name} já existe")
            continue
        old.rename(new)
        print(f"  ✓ {old.name} → {new.name}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
