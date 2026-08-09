"""Web subprocess entry points for the existing pipeline modules."""

from __future__ import annotations

import argparse
from pathlib import Path


def _disable_numba_disk_cache() -> None:
    """Keep JIT enabled while avoiding broken cache locators in web subprocesses."""

    try:
        import numba
        import numba.core.decorators
    except Exception:
        return

    real_jit = numba.jit

    def jit_without_cache(*args, **kwargs):
        kwargs["cache"] = False
        return real_jit(*args, **kwargs)

    numba.jit = jit_without_cache
    numba.core.decorators.jit = jit_without_cache

    real_guvectorize = getattr(numba, "guvectorize", None)
    if real_guvectorize is not None:

        def guvectorize_without_cache(*args, **kwargs):
            kwargs["cache"] = False
            return real_guvectorize(*args, **kwargs)

        numba.guvectorize = guvectorize_without_cache

    real_vectorize = getattr(numba, "vectorize", None)
    if real_vectorize is not None:

        def vectorize_without_cache(*args, **kwargs):
            kwargs["cache"] = False
            return real_vectorize(*args, **kwargs)

        numba.vectorize = vectorize_without_cache


def run_diarization(args: argparse.Namespace) -> None:
    _disable_numba_disk_cache()

    from Project.run_diarization import run_pipeline

    run_pipeline(
        args.input,
        output_dir=args.output_dir,
        reference_rttm=args.reference_rttm,
        device=args.device,
        backend=args.backend,
        num_speakers=args.num_speakers,
        oracle_num_speakers=args.oracle_num_speakers,
        save_json=args.save_json,
        save_csv=args.save_csv,
        save_rttm=args.save_rttm,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run web-safe pipeline subprocesses.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diarization = subparsers.add_parser("diarization")
    diarization.add_argument("--input", required=True, type=Path)
    diarization.add_argument("--output-dir", required=True, type=Path)
    diarization.add_argument("--reference-rttm", type=Path, default=None)
    diarization.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    diarization.add_argument("--backend", choices=["auto", "pyannote", "webrtc", "energy"], default=None)
    diarization.add_argument("--num-speakers", type=int, default=None)
    diarization.add_argument("--oracle-num-speakers", action=argparse.BooleanOptionalAction, default=None)
    diarization.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=True)
    diarization.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=True)
    diarization.add_argument("--save-rttm", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "diarization":
        run_diarization(args)


if __name__ == "__main__":
    main()
