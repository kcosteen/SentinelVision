"""
Unit tests for the Roboflow importer.

The network call itself can't be tested without a real key, so what's pinned down
here is everything around it: the zip is unpacked correctly, a malicious archive
can't write outside its destination, and a missing key fails with an explanation
rather than a traceback.

`download_and_extract` is exercised through a `file://` URL -- urllib treats it
like any other, so the streaming and extraction paths are the real ones.
"""

import os
import pathlib
import zipfile

import pytest

from src.detection.roboflow_import import (
    API_KEY_ENV,
    download_and_extract,
    require_api_key,
)


def make_zip(path, entries):
    """Write a zip containing {archive_name: text}."""
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in entries.items():
            bundle.writestr(name, content)
    return path


def as_file_url(path):
    return pathlib.Path(path).absolute().as_uri()


def test_require_api_key_reads_the_environment(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "abc123")
    assert require_api_key() == "abc123"


def test_require_api_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "  abc123\n")
    assert require_api_key() == "abc123"


def test_missing_api_key_explains_how_to_set_it(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(SystemExit) as error:
        require_api_key()
    message = str(error.value)
    assert API_KEY_ENV in message
    assert "roboflow.com" in message


def test_blank_api_key_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "   ")
    with pytest.raises(SystemExit):
        require_api_key()


def test_download_and_extract_unpacks_the_archive(tmp_path):
    archive = make_zip(tmp_path / "export.zip", {
        "data.yaml": "names: ['phone']\nnc: 1\n",
        "train/images/a.jpg": "not-really-a-jpeg",
        "train/labels/a.txt": "0 0.5 0.5 0.2 0.2\n",
    })
    destination = tmp_path / "out"

    download_and_extract(as_file_url(archive), str(destination))

    assert (destination / "data.yaml").exists()
    assert (destination / "train" / "images" / "a.jpg").exists()
    assert (destination / "train" / "labels" / "a.txt").exists()


def test_the_downloaded_zip_is_cleaned_up(tmp_path):
    archive = make_zip(tmp_path / "export.zip", {"data.yaml": "names: ['phone']\n"})
    destination = tmp_path / "out"

    download_and_extract(as_file_url(archive), str(destination))

    # The intermediate archive shouldn't be left behind inside the dataset.
    assert not (destination / "_export.zip").exists()


def test_a_zip_that_escapes_its_destination_is_refused(tmp_path):
    """Zip-slip: an entry like '../evil.txt' must not write outside destination."""
    archive = make_zip(tmp_path / "evil.zip", {"../escaped.txt": "pwned"})
    destination = tmp_path / "out"

    with pytest.raises(SystemExit) as error:
        download_and_extract(as_file_url(archive), str(destination))
    assert "unsafe path" in str(error.value)

    assert not (tmp_path / "escaped.txt").exists()


def test_absolute_paths_in_a_zip_are_refused(tmp_path):
    archive = make_zip(tmp_path / "evil.zip", {"/tmp/escaped.txt": "pwned"})
    destination = tmp_path / "out"

    with pytest.raises(SystemExit):
        download_and_extract(as_file_url(archive), str(destination))


def test_extraction_is_idempotent_for_the_same_archive(tmp_path):
    archive = make_zip(tmp_path / "export.zip", {"data.yaml": "names: ['phone']\n"})
    destination = tmp_path / "out"

    download_and_extract(as_file_url(archive), str(destination))
    download_and_extract(as_file_url(archive), str(destination))

    assert (destination / "data.yaml").read_text().startswith("names:")
    assert len(os.listdir(destination)) == 1
