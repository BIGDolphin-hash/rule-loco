#!/usr/bin/env python3
"""Repair and relock one class's Global Patch bank as a transaction.

Unlocks the bank and its shard store, repairs and verifies the bank, and
attempts to restore locks even when repair fails.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

GLOBAL_SUFFIX = ".dino_full_coreset05_b6b12b18b24.npz"
CLASS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class TransactionError(RuntimeError):
    """A repair transaction could not be completed safely."""


def _read_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"].item()))


def _resolve_store(bank: Path, metadata_key: str) -> Path:
    metadata = _read_metadata(bank)
    store_name = metadata.get(metadata_key)
    if (
        not isinstance(store_name, str)
        or not store_name
        or Path(store_name).name != store_name
    ):
        raise TransactionError(
            f"{bank.name} has an invalid {metadata_key}: {store_name!r}"
        )
    store = (bank.parent / store_name).resolve()
    if store.parent != bank.parent.resolve() or not store.is_dir():
        raise TransactionError(f"texture shard store not found: {store}")
    return store


def _run(
    command: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None
) -> None:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        raise TransactionError(
            f"command failed with exit status {completed.returncode}: "
            + " ".join(command)
        )


def _privileged_command(command: Sequence[str]) -> list[str]:
    if os.geteuid() == 0:
        return list(command)
    return ["sudo", *command]


def _set_write_permission(paths: Sequence[Path], *, writable: bool, cwd: Path) -> None:
    mode = "u+w" if writable else "a-w"
    for path in paths:
        recursive = path.is_dir()
        command = ["chmod"]
        if recursive:
            command.append("-R")
        command.extend((mode, str(path)))
        _run(command, cwd=cwd)


def _set_immutable(paths: Sequence[Path], *, immutable: bool, cwd: Path) -> None:
    operation = "+i" if immutable else "-i"
    for path in paths:
        command = ["chattr"]
        if path.is_dir():
            command.append("-R")
        command.extend((operation, str(path)))
        _run(_privileged_command(command), cwd=cwd)


def _immutable_flag(path: Path, *, cwd: Path) -> bool:
    completed = subprocess.run(
        ["lsattr", "-d", str(path)],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return False
    fields = completed.stdout.split()
    return bool(fields and "i" in fields[0])


def _first_file(directory: Path) -> Path:
    for root, _, filenames in os.walk(directory):
        if filenames:
            return Path(root) / sorted(filenames)[0]
    raise TransactionError(f"texture shard store is empty: {directory}")


def _verify_locked(paths: Sequence[Path], stores: Sequence[Path], *, cwd: Path) -> None:
    targets = [*paths, *stores, *(_first_file(store) for store in stores)]
    missing = [str(path) for path in targets if not _immutable_flag(path, cwd=cwd)]
    if missing:
        raise TransactionError(
            "immutable lock verification failed: " + ", ".join(missing)
        )
    print("LOCK VERIFIED", flush=True)
    for path in targets:
        subprocess.run(["lsattr", "-d", str(path)], cwd=cwd, check=False)


def _lock_banks(
    banks: Sequence[Path],
    stores: Sequence[Path],
    *,
    cwd: Path,
) -> None:
    """Attempt every locking step before reporting any individual failure."""

    errors: list[str] = []
    for path in (*stores, *banks):
        try:
            _set_write_permission((path,), writable=False, cwd=cwd)
        except Exception as exc:
            errors.append(f"chmod {path}: {exc}")
    for path in (*stores, *banks):
        try:
            _set_immutable((path,), immutable=True, cwd=cwd)
        except Exception as exc:
            errors.append(f"chattr {path}: {exc}")
    try:
        _verify_locked(banks, stores, cwd=cwd)
    except Exception as exc:
        errors.append(str(exc))
    if errors:
        raise TransactionError("; ".join(errors))


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _checker_command(
    *,
    project: Path,
    root: Path,
    class_name: str,
    device: str,
    repair: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(project / "scripts/check_repair_texture_banks.py"),
        "--root",
        str(root),
        "--scope",
        "global",
        "--global-pattern",
        f"{class_name}{GLOBAL_SUFFIX}",
        "--device",
        device,
    ]
    if repair:
        command.append("--repair")
    return command


def _install_signal_handlers() -> None:
    def interrupt(signum: int, _frame: object) -> None:
        raise InterruptedError(f"received signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)


def repair_lock_class(
    *,
    project: Path,
    root: Path,
    class_name: str,
    device: str,
    verify_passes: int,
) -> None:
    if CLASS_NAME_PATTERN.fullmatch(class_name) is None:
        raise TransactionError(f"invalid class name: {class_name!r}")
    if verify_passes < 1:
        raise TransactionError("verify-passes must be at least 1")

    project = project.expanduser().resolve()
    root = root.expanduser().resolve()
    global_bank = root / "global" / f"{class_name}{GLOBAL_SUFFIX}"
    banks = (global_bank,)
    for bank in banks:
        if not bank.is_file():
            raise TransactionError(f"texture memory bank not found: {bank}")

    global_store = _resolve_store(global_bank, "image_shard_store")
    stores = (global_store,)
    lock_targets = (*stores, *banks)
    state_path = root / f".texture-repair-{class_name}.json"
    process_lock_path = root / ".texture-repair.lock"
    process_lock_path.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    source_path = str(project / "src")
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not old_pythonpath else f"{source_path}:{old_pythonpath}"
    )

    with process_lock_path.open("a+", encoding="utf-8") as process_lock:
        try:
            fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TransactionError(
                "another texture repair transaction is running"
            ) from exc

        if os.geteuid() != 0:
            _run(["sudo", "-v"], cwd=project)

        _write_state(
            state_path,
            {
                "class_name": class_name,
                "phase": "preparing",
                "banks": [str(path) for path in banks],
                "stores": [str(path) for path in stores],
            },
        )
        needs_relock = True
        locked = False
        try:
            print(f"UNLOCK {class_name}", flush=True)
            _set_immutable(lock_targets, immutable=False, cwd=project)
            _set_write_permission(stores, writable=True, cwd=project)
            _set_write_permission(banks, writable=True, cwd=project)
            _write_state(
                state_path,
                {
                    "class_name": class_name,
                    "phase": "repairing",
                    "banks": [str(path) for path in banks],
                    "stores": [str(path) for path in stores],
                },
            )

            print(f"REPAIR {class_name}", flush=True)
            _run(
                _checker_command(
                    project=project,
                    root=root,
                    class_name=class_name,
                    device=device,
                    repair=True,
                ),
                cwd=project,
                env=environment,
            )
            os.sync()

            for pass_index in range(1, verify_passes + 1):
                print(
                    f"VERIFY {class_name} pass {pass_index}/{verify_passes}",
                    flush=True,
                )
                _run(
                    _checker_command(
                        project=project,
                        root=root,
                        class_name=class_name,
                        device=device,
                        repair=False,
                    ),
                    cwd=project,
                    env=environment,
                )

            print(f"LOCK {class_name}", flush=True)
            _lock_banks(banks, stores, cwd=project)
            locked = True

            print(f"POST-LOCK VERIFY {class_name}", flush=True)
            _run(
                _checker_command(
                    project=project,
                    root=root,
                    class_name=class_name,
                    device=device,
                    repair=False,
                ),
                cwd=project,
                env=environment,
            )
            needs_relock = False
            try:
                state_path.unlink()
            except FileNotFoundError:
                pass
            print(f"TRANSACTION COMPLETE: {class_name}", flush=True)
        finally:
            if needs_relock:
                print(
                    f"FAIL-CLOSED RELOCK {class_name}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    _lock_banks(banks, stores, cwd=project)
                    locked = True
                except Exception as relock_error:
                    print(
                        f"CRITICAL: relock failed: {relock_error}",
                        file=sys.stderr,
                        flush=True,
                    )
                if locked:
                    try:
                        state_path.unlink()
                    except FileNotFoundError:
                        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "unlock, repair, repeatedly verify and immutably relock one class's "
            "Global Patch texture banks"
        )
    )
    parser.add_argument("--class-name", required=True)
    parser.add_argument(
        "--root",
        default="memory_banks/Global patch/full/mvtec_loco",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-passes", type=int, default=2)
    args = parser.parse_args()

    _install_signal_handlers()
    project = Path(__file__).resolve().parents[1]
    root = Path(args.root).expanduser()
    if not root.is_absolute():
        root = project / root
    try:
        repair_lock_class(
            project=project,
            root=root,
            class_name=args.class_name,
            device=args.device,
            verify_passes=args.verify_passes,
        )
    except (KeyboardInterrupt, InterruptedError) as exc:
        print(f"INTERRUPTED: {exc}", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
