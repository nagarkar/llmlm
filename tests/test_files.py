"""Atomic writes (`llmlm.files.write_atomic`): a crash leaves the old file or the new one,
never a torn one; a secret's mode is set before its bytes land."""

import os
import stat

import pytest

from llmlm import files


def test_write_atomic_replaces_the_file_and_leaves_no_temp(tmp_path):
    target = tmp_path / "a" / "note.md"
    files.write_atomic(target, "one")
    files.write_atomic(target, b"two")
    assert target.read_text() == "two"
    assert [path.name for path in target.parent.iterdir()] == ["note.md"]


def test_a_failed_write_leaves_the_old_file_whole(tmp_path, monkeypatch):
    target = tmp_path / "note.md"
    target.write_text("old")

    def broken(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError):
        files.write_atomic(target, "new")
    assert target.read_text() == "old"
    assert [path.name for path in tmp_path.iterdir()] == ["note.md"]


def test_an_existing_file_keeps_its_mode_and_a_given_mode_wins(tmp_path):
    target = tmp_path / "secret"
    files.write_atomic(target, "a", mode=0o600)
    files.write_atomic(target, "b")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    files.write_atomic(target, "c", mode=0o640)
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
