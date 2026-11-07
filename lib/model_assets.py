"""Install checksum-pinned model files atomically in a caller-owned cache."""

import hashlib
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


@dataclass(frozen=True)
class AssetSpec:
    path: str
    is_local: bool
    sha256: str
    size_bytes: Optional[int] = None
    max_bytes: int = 1024 * 1024 * 1024


def asset_spec(source, *, size_bytes=None, max_bytes=1024 * 1024 * 1024, base_dir=None):
    if (not isinstance(source, dict) or type(source.get("is_local")) is not bool
            or not isinstance(source.get("path"), str) or not source["path"].strip()):
        raise ValueError("Model assets require path, boolean is_local and sha256")
    digest = str(source.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Model assets require a pinned SHA-256")
    count = size_bytes if size_bytes is not None else source.get("size_bytes", source.get("bytes"))
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError("Model asset size_bytes must be a positive integer")
    if type(max_bytes) is not int or max_bytes <= 0 or (count is not None and count > max_bytes):
        raise ValueError("Model asset exceeds its configured maximum size")
    path = source["path"].strip()
    if source["is_local"]:
        local = Path(path).expanduser()
        if not local.is_absolute() and base_dir is not None:
            local = Path(base_dir) / local
        path = str(local)
    elif not path.startswith("https://"):
        raise ValueError("Remote model assets require HTTPS")
    return AssetSpec(path, source["is_local"], digest, count, max_bytes)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path, spec):
    path = Path(path)
    if not path.is_file():
        return False
    count = path.stat().st_size
    return (0 < count <= spec.max_bytes and (spec.size_bytes is None or count == spec.size_bytes)
            and file_sha256(path) == spec.sha256)


@contextmanager
def _asset_lock(path):
    with Path(path).open("a+b") as handle:
        try:
            import fcntl
        except ImportError:
            yield
        else:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_asset(spec, cache_dir, *, suffix=".bin", timeout=180):
    """Verify a cached file or publish a newly verified local/HTTPS copy."""
    if not re.fullmatch(r"\.[a-zA-Z0-9]+", suffix):
        raise ValueError("Model asset suffix must be a simple file extension")
    cache = Path(cache_dir).expanduser()
    temporary = None
    try:
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / (spec.sha256 + suffix)
        with _asset_lock(cache / (spec.sha256 + ".lock")):
            if verify_asset(destination, spec):
                return destination
            digest = hashlib.sha256()
            count = 0

            def copy_chunks(chunks, handle):
                nonlocal count
                for chunk in chunks:
                    if not chunk:
                        continue
                    count += len(chunk)
                    if count > spec.max_bytes or (spec.size_bytes is not None and count > spec.size_bytes):
                        raise ValueError("Model asset exceeds its pinned or maximum byte size")
                    digest.update(chunk)
                    handle.write(chunk)

            with tempfile.NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                if spec.is_local:
                    with Path(spec.path).open("rb") as source:
                        copy_chunks(iter(lambda: source.read(1024 * 1024), b""), handle)
                else:
                    with requests.get(spec.path, stream=True, timeout=(10, timeout)) as response:
                        response.raise_for_status()
                        copy_chunks(response.iter_content(1024 * 1024), handle)
                handle.flush()
                os.fsync(handle.fileno())
            if (count == 0 or digest.hexdigest() != spec.sha256
                    or (spec.size_bytes is not None and count != spec.size_bytes)):
                raise ValueError("Model asset failed SHA-256 or byte-size verification")
            temporary.chmod(0o644)
            os.replace(temporary, destination)
            return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
