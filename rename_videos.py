#!/usr/bin/env python3
"""
rename_videos.py — renomeia vídeos automaticamente com data + slug gerado por IA.

Fluxo por arquivo:
  1. Descobre a data de gravação (em ordem de confiabilidade):
       a) metadata do arquivo via ffprobe (creation_time) — funciona com
          qualquer câmera: DJI, GoPro, Sony, Canon, celular, etc.
       b) padrões conhecidos no nome do arquivo (DJI, Samsung, ISO, etc.)
       c) data de modificação do arquivo (último recurso)
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
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".mts", ".m2ts", ".mxf", ".wmv"}

# ---------------------------------------------------------------------------
# Extração de data — três camadas em ordem decrescente de confiabilidade
# ---------------------------------------------------------------------------

# Padrões de nome de arquivo com data embutida.
# Usados apenas quando o metadata do container não tem creation_time.
# Cada tupla: (regex, índices dos grupos year/month/day)
_FILENAME_PATTERNS: list[tuple[re.Pattern, tuple[int, int, int]]] = [
    # DJI:      DJI_20260501134401_0039_D.ext
    (re.compile(r"DJI_(\d{4})(\d{2})(\d{2})\d+", re.IGNORECASE), (1, 2, 3)),
    # Samsung/Android:  VID_20260501_134401.ext
    (re.compile(r"VID_(\d{4})(\d{2})(\d{2})_\d+", re.IGNORECASE), (1, 2, 3)),
    # WhatsApp:  VID-20260501-WA0001.ext
    (re.compile(r"VID-(\d{4})(\d{2})(\d{2})-", re.IGNORECASE), (1, 2, 3)),
    # OBS / screen capture:  2026-05-01 13-44-01.mkv  ou  2026-05-01_13-44.ext
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[ _T\-]"), (1, 2, 3)),
    # GoPro com data:  GOPR20260501.ext  (raro, mas existe)
    (re.compile(r"GOPR(\d{4})(\d{2})(\d{2})", re.IGNORECASE), (1, 2, 3)),
    # Genérico YYYYMMDD em qualquer posição do nome
    (re.compile(r"(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])"), (1, 2, 3)),
]


def _validate_date(year: str, month: str, day: str) -> bool:
    y, m, d = int(year), int(month), int(day)
    return 2000 <= y <= 2099 and 1 <= m <= 12 and 1 <= d <= 31


def _date_from_ffprobe(path: Path) -> str | None:
    """Lê creation_time do container via ffprobe. Funciona com qualquer câmera."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "0",
             "-show_entries", "format_tags=creation_time",
             "-of", "json", str(path)],
            stderr=subprocess.DEVNULL,
        ).decode()
        ct = json.loads(out).get("format", {}).get("tags", {}).get("creation_time", "")
        if not ct:
            return None
        # Formato: "2026-05-01T13:44:01.000000Z"
        dt = datetime.fromisoformat(ct.rstrip("Z").split(".")[0])
        return dt.strftime("%d-%m-%Y")
    except Exception:
        return None


def _date_from_filename(stem: str) -> str | None:
    """Tenta extrair data do nome do arquivo usando padrões conhecidos."""
    for pattern, (yi, mi, di) in _FILENAME_PATTERNS:
        m = pattern.search(stem)
        if m and _validate_date(m.group(yi), m.group(mi), m.group(di)):
            return f"{m.group(di)}-{m.group(mi)}-{m.group(yi)}"
    return None


def parse_date(path: Path) -> tuple[str, str]:
    """Retorna (data DD-MM-YYYY, fonte) para o vídeo."""
    d = _date_from_ffprobe(path)
    if d:
        return d, "metadata"
    d = _date_from_filename(path.stem)
    if d:
        return d, "nome"
    d = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d-%m-%Y")
    return d, "mtime"


SLUG_PROMPT = """\
Você receberá a transcrição de um trecho de vlog em português brasileiro.
Gere um slug com 4 a 6 palavras que capture o tema principal ou o momento \
mais marcante. Use kebab-case (sem acentos, tudo minúsculo, palavras \
separadas por hífen). Seja específico — evite generalizações como \
"falando-sobre-algo" ou "video-do-dia". Responda APENAS com o slug.

Exemplos válidos:
  chegou-minha-nova-camera-dji-osmo
  cortei-o-cabelo-pela-primeira-vez
  dicas-de-edicao-que-uso-todo-dia
  comprando-roupa-nova-com-minha-mae
  mudei-de-cidade-e-to-nervosa

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


# ---------------------------------------------------------------------------
# Transcrição (faster-whisper, local, sem API)
# ---------------------------------------------------------------------------

def _load_whisper(model_name: str) -> "WhisperModel":
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Erro: faster-whisper não instalado. Execute: pip install faster-whisper")

    import ctranslate2, threading

    try:
        device, compute = ("cuda", "float16") if ctranslate2.get_cuda_device_count() > 0 else ("cpu", "int8")
    except Exception:
        device, compute = "cpu", "int8"

    holder: list = [None]
    done = threading.Event()

    def _load():
        holder[0] = WhisperModel(model_name, device=device, compute_type=compute)
        done.set()

    threading.Thread(target=_load, daemon=True).start()

    spinner = ['⠋', '⠙', '⠸', '⠼', '⠴', '⠦', '⠇', '⠏']
    i = 0
    while not done.wait(0.12):
        print(f"\r  whisper: carregando {model_name} ({device})... {spinner[i % len(spinner)]}",
              end="", flush=True)
        i += 1
    print(f"\r  whisper: {model_name} ({device}) pronto                    ")

    return holder[0]


def transcribe(video: Path, model: "WhisperModel", max_seconds: float) -> str:
    """Transcreve até max_seconds do vídeo. Retorna o texto completo."""
    import threading

    parts: list[str] = []
    pos = [0.0]       # posição atual em segundos (atualizada pelo worker)
    err: list[Exception | None] = [None]
    done = threading.Event()

    def _worker() -> None:
        try:
            segs, _ = model.transcribe(str(video), word_timestamps=False)
            for seg in segs:
                parts.append(seg.text.strip())
                pos[0] = min(seg.end, max_seconds)
                if seg.end >= max_seconds:
                    break
        except Exception as exc:
            err[0] = exc
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()

    # barra de progresso no thread principal — atualiza a cada 120ms
    spinner = ['⠋', '⠙', '⠸', '⠼', '⠴', '⠦', '⠇', '⠏']
    i = 0
    while not done.wait(0.12):
        p = pos[0]
        filled = int(p / max_seconds * 28)
        bar = '█' * filled + '░' * (28 - filled)
        print(f"\r  [{bar}] {p:.0f}/{max_seconds:.0f}s {spinner[i % len(spinner)]}",
              end="", flush=True)
        i += 1
    print(f"\r  [{'█' * 28}] {max_seconds:.0f}/{max_seconds:.0f}s   ", flush=True)

    if err[0] is not None:
        if isinstance(err[0], IndexError):
            print("  aviso  : sem áudio ou erro ao decodificar (IndexError)")
            return ""
        raise err[0]

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


def _is_broll(transcript: str) -> bool:
    """Detecta b-roll/sem fala por repetição excessiva — sinal de alucinação do whisper."""
    if not transcript:
        return True
    words = transcript.lower().split()
    if len(words) < 4:
        return True
    # Proporção baixa de palavras únicas (ex: "thank you thank you thank you")
    if len(set(words)) / len(words) < 0.55:
        return True
    # Bigrama dominante muito repetido (ex: "a cidade no brasil a cidade no brasil")
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    top_bi = max(set(bigrams), key=bigrams.count)
    if bigrams.count(top_bi) >= 2 and bigrams.count(top_bi) / len(bigrams) > 0.35:
        return True
    # Frases repetidas (ex: "The car is on the car. The car is on the car.")
    sentences = [s.strip() for s in re.split(r'[.!?]', transcript.lower()) if len(s.strip()) > 8]
    if len(sentences) >= 3 and len(set(sentences)) / len(sentences) < 0.65:
        return True
    return False


def generate_slug(transcript: str, provider: str, ollama_model: str, ai_cmd: str) -> str:
    if _is_broll(transcript):
        return "broll"
    if provider == "anthropic":
        return _slug_anthropic(transcript)
    if provider == "ollama":
        return _slug_ollama(transcript, ollama_model)
    if provider == "cmd":
        if not ai_cmd:
            sys.exit("Erro: --ai-cmd obrigatório quando --provider=cmd")
        return _slug_cmd(transcript, ai_cmd)
    raise ValueError(f"Provider desconhecido: {provider}")


def _check_provider_deps(provider: str) -> None:
    if provider == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            sys.exit("Erro: pacote 'anthropic' não instalado. Execute: pip install anthropic")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Renomeia vídeos com data + slug gerado por IA via transcrição.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", nargs="?", default=".", help="Diretório ou arquivo de vídeo (padrão: pasta atual)")
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
    ap.add_argument("--max-seconds", type=float, default=180.0,
                    help="Segundos máximos de áudio a transcrever por arquivo (padrão: 180)")
    ap.add_argument("--cache-dir", default=None,
                    help="Pasta para cache de transcrições (padrão: .rename_cache/ ao lado dos vídeos)")
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

    video_dir = target if target.is_dir() else target.parent
    cache_dir = Path(args.cache_dir) if args.cache_dir else video_dir / ".rename_cache"
    cache_dir.mkdir(exist_ok=True)

    _check_provider_deps(args.provider)

    print(f"{len(videos)} vídeo(s) encontrado(s) | IA: {args.provider} | "
          f"whisper: {args.whisper_model} | max: {args.max_seconds:.0f}s\n")

    whisper_model = _load_whisper(args.whisper_model)

    renames: list[tuple[Path, Path]] = []

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")

        date_str, date_src = parse_date(video)
        print(f"  data   : {date_str}  (via {date_src})")

        cache_file = cache_dir / f"{video.stem}.txt"
        if cache_file.exists():
            transcript = cache_file.read_text(encoding="utf-8")
            print(f"  whisper: usando cache ({cache_file.name})")
        else:
            print(f"  whisper: transcrevendo primeiros {args.max_seconds:.0f}s...")
            transcript = transcribe(video, whisper_model, args.max_seconds)
            cache_file.write_text(transcript, encoding="utf-8")

        preview = (transcript[:90] + "...") if len(transcript) > 90 else transcript
        broll = _is_broll(transcript)
        print(f"  texto  : {preview or '(vazio)'}{'  [b-roll]' if broll else ''}")

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
