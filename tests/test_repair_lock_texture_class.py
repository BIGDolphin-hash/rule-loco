from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/repair_lock_texture_class.py"
SPEC = importlib.util.spec_from_file_location("repair_lock_texture_class", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RepairLockTextureClassTests(unittest.TestCase):
    def make_layout(self, root: Path, class_name: str = "demo") -> None:
        global_root = root / "global"
        global_root.mkdir(parents=True)
        global_store = global_root / "global.images.test"
        global_store.mkdir()
        (global_store / "shard.npz").write_bytes(b"global")
        np.savez(
            global_root / f"{class_name}{MODULE.GLOBAL_SUFFIX}",
            metadata=np.asarray(json.dumps({"image_shard_store": global_store.name})),
        )

    def test_failed_repair_always_attempts_relock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "banks"
            project = Path(directory) / "project"
            project.mkdir()
            self.make_layout(root)
            immutable_calls: list[bool] = []
            permission_calls: list[bool] = []

            def fail_repair(command, **kwargs):
                if "--repair" in command:
                    raise MODULE.TransactionError("simulated repair failure")

            with (
                patch.object(MODULE, "_run", side_effect=fail_repair),
                patch.object(
                    MODULE,
                    "_set_immutable",
                    side_effect=lambda paths, *, immutable, cwd: immutable_calls.append(
                        immutable
                    ),
                ),
                patch.object(
                    MODULE,
                    "_set_write_permission",
                    side_effect=lambda paths, *, writable, cwd: permission_calls.append(
                        writable
                    ),
                ),
                patch.object(MODULE, "_verify_locked"),
                patch.object(MODULE.os, "geteuid", return_value=0),
            ):
                with self.assertRaisesRegex(
                    MODULE.TransactionError, "simulated repair failure"
                ):
                    MODULE.repair_lock_class(
                        project=project,
                        root=root,
                        class_name="demo",
                        device="cpu",
                        verify_passes=2,
                    )

            self.assertIn(False, immutable_calls)
            self.assertTrue(immutable_calls[-2:] == [True] * 2)
            self.assertIn(True, permission_calls)
            self.assertTrue(permission_calls[-2:] == [False] * 2)

    def test_success_runs_repair_two_verifications_and_post_lock_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "banks"
            project = Path(directory) / "project"
            project.mkdir()
            self.make_layout(root)
            commands: list[list[str]] = []

            with (
                patch.object(
                    MODULE,
                    "_run",
                    side_effect=lambda command, **kwargs: commands.append(
                        list(command)
                    ),
                ),
                patch.object(MODULE, "_set_immutable"),
                patch.object(MODULE, "_set_write_permission"),
                patch.object(MODULE, "_verify_locked"),
                patch.object(MODULE.os, "geteuid", return_value=0),
                patch.object(MODULE.os, "sync"),
            ):
                MODULE.repair_lock_class(
                    project=project,
                    root=root,
                    class_name="demo",
                    device="cpu",
                    verify_passes=2,
                )

            checker_commands = [
                command
                for command in commands
                if any("check_repair_texture_banks.py" in value for value in command)
            ]
            self.assertEqual(len(checker_commands), 4)
            self.assertIn("--repair", checker_commands[0])
            self.assertTrue(
                all("--repair" not in command for command in checker_commands[1:])
            )
            self.assertFalse((root / ".texture-repair-demo.json").exists())

    def test_lock_attempts_chattr_even_when_chmod_fails(self) -> None:
        paths = tuple(Path(f"/tmp/target-{index}") for index in range(4))
        immutable_calls: list[Path] = []

        with (
            patch.object(
                MODULE,
                "_set_write_permission",
                side_effect=MODULE.TransactionError("simulated chmod failure"),
            ),
            patch.object(
                MODULE,
                "_set_immutable",
                side_effect=lambda items, *, immutable, cwd: immutable_calls.extend(
                    items
                ),
            ),
            patch.object(MODULE, "_verify_locked"),
        ):
            with self.assertRaisesRegex(
                MODULE.TransactionError, "simulated chmod failure"
            ):
                MODULE._lock_banks(paths[:2], paths[2:], cwd=Path("/tmp"))

        self.assertEqual(immutable_calls, [*paths[2:], *paths[:2]])


if __name__ == "__main__":
    unittest.main()
