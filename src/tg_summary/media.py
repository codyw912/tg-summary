"""Local media handling for the media and link stage: hashing, ffmpeg conversions, and speech-to-text requests.

Transcription uses the OpenAI-compatible `/audio/transcriptions` endpoint, which whisper.cpp's server,
mlx-audio's server, oMLX, vLLM and OpenAI all serve. Audio is converted to 16 kHz mono WAV first, because
that is the one format every one of those servers accepts.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx2

from tg_summary.config import Role
from tg_summary.store import Store

FRAME_WIDTH = 1024  # key frames are scaled down to at most this width


class MediaError(RuntimeError):
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def inside(root: Path, relative: str) -> Path | None:
    """The file `relative` names under `root`, or None if it is missing or resolves outside `root`.

    Resolving follows symlinks, so a link inside the export that points elsewhere is refused. Nothing outside
    the export may be read, hashed or uploaded, whatever path a (possibly crafted) export contains."""
    base = root.resolve()
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        return None
    return candidate


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def media_type(path: Path, mime: str | None) -> str:
    return mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def to_wav(path: Path) -> bytes | None:
    """16 kHz mono WAV of the file's first audio stream, or None when the file has no audio."""
    if not _probe(path, "-select_streams", "a", "-show_entries", "stream=index").get("streams"):
        return None
    # Written to a file rather than a pipe, so ffmpeg can fill in the header's length fields.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "audio.wav"
        _ffmpeg(["-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(out)])
        return out.read_bytes()


def key_frames(path: Path, count: int, duration: float | None) -> list[bytes]:
    """`count` JPEG frames spread evenly over the video (one frame for very short clips)."""
    seconds = duration if duration else probe_duration(path)
    times = [seconds * (i + 0.5) / count for i in range(count)] if seconds and seconds > 1 else [0.0]
    frames = []
    for t in times:
        frame = _ffmpeg(
            [
                "-ss",
                f"{t:.2f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                f"scale='min({FRAME_WIDTH},iw)':-2",
                "-f",
                "image2pipe",
                "-c:v",
                "mjpeg",
                "-",
            ]
        )
        if frame:
            frames.append(frame)
    if not frames:
        raise MediaError(f"no frames could be extracted from {path.name}", "no_frames")
    return frames


def probe_duration(path: Path) -> float | None:
    try:
        return float(_probe(path, "-show_entries", "format=duration")["format"]["duration"])
    except (ValueError, KeyError, TypeError):
        return None


def _probe(path: Path, *args: str) -> dict:
    if shutil.which("ffprobe") is None:
        raise MediaError("ffprobe is not installed", "ffmpeg_missing")
    done = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "json", str(path)], capture_output=True, check=False)
    try:
        return json.loads(done.stdout) or {}
    except ValueError:
        return {}


def transcribe(role: Role, store: Store, wav: bytes, item: str) -> str:
    """Text of one WAV file from `{base_url}/audio/transcriptions`.

    Calls to `external` endpoints are audited with the request's form fields and the file's size and hash in
    place of its bytes, plus the exact response body.
    """
    ep = role.endpoint
    url = f"{(ep.base_url or '').rstrip('/')}/audio/transcriptions"
    key = os.environ.get(ep.api_key_env) if ep.api_key_env else None
    if ep.api_key_env and not key:
        raise MediaError(f"environment variable {ep.api_key_env} is not set (endpoint {ep.name})", "missing_api_key")
    headers = {**role.extra_headers, **({"Authorization": f"Bearer {key}"} if key else {})}
    fields = {"model": role.model, **{k: str(v) for k, v in role.extra_body.items()}}
    with httpx2.Client(timeout=ep.timeout_seconds, trust_env=False) as client:
        try:
            response = client.post(url, data=fields, files={"file": ("audio.wav", wav, "audio/wav")}, headers=headers)
        except httpx2.HTTPError as e:
            raise MediaError(f"{ep.name}: {type(e).__name__}", "connection_error") from e
    if ep.trust == "external":
        sent = {"url": url, "fields": fields, "file": {"bytes": len(wav), "item": item}}
        store.audit(ep.name, role.model, json.dumps([sent]), json.dumps([response.text], ensure_ascii=False), [])
    if response.status_code != 200:
        raise MediaError(f"{ep.name} returned {response.status_code}", f"http_{response.status_code}")
    try:
        return " ".join(str(response.json()["text"]).split())
    except (ValueError, KeyError, TypeError) as e:
        raise MediaError(f"{ep.name} returned no transcript", "unusable_output") from e


def _ffmpeg(args: list[str]) -> bytes:
    if shutil.which("ffmpeg") is None:
        raise MediaError("ffmpeg is not installed", "ffmpeg_missing")
    done = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *args], capture_output=True, check=False)
    if done.returncode != 0:
        # No message text is involved here; stderr names only the file and the codec problem.
        raise MediaError(done.stderr.decode(errors="replace")[-300:], "ffmpeg_failed")
    return done.stdout
