"""Filesystem and failed-download regressions; no network or models required."""
import io
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

import langswap.file_repository as storage
from langswap.pipeline_models.models import RemoteFile


@pytest.fixture(params=[storage.LocalOnlyFileRepository, storage.RemoteFileRepository, storage.LocalFileRepository])
def repo(request, tmp_path):
    if request.param is storage.LocalOnlyFileRepository:
        return request.param('job', str(tmp_path))
    return request.param('job', str(tmp_path), Mock())


@pytest.mark.parametrize('name', ['../outside', '/outside', 'C:\\outside', '..\\outside', 'sub/../../outside', 'x:stream', '.'])
def test_reject_paths_outside_job(repo, name):
    with pytest.raises(ValueError):
        repo.get_file(name)
    with pytest.raises(ValueError):
        repo.subdir(name)
    with pytest.raises(ValueError):
        repo.materialize_file(RemoteFile(name=name, s3_url='https://example.invalid/input'))


@pytest.mark.parametrize('public_id', ['../bad', '/bad', 'x/y', 'x\\y', 'C:\\bad', None, ''])
def test_reject_unsafe_job_ids(tmp_path, public_id):
    with pytest.raises(ValueError):
        storage.LocalOnlyFileRepository(public_id, str(tmp_path))


def test_nested_file_and_local_file_uri(tmp_path):
    repo = storage.LocalOnlyFileRepository('job', str(tmp_path))
    saved = repo.save_file_from_stream(repo.get_file('sub/hello world.txt'), io.BytesIO(b'content'))
    assert Path(saved.file_path).read_bytes() == b'content'
    assert saved.s3_url == Path(saved.file_path).as_uri()


def test_local_save_cannot_claim_missing_artifact(tmp_path):
    repo = storage.LocalOnlyFileRepository('job', str(tmp_path))
    with pytest.raises(FileNotFoundError):
        repo.save_file(repo.get_file('missing.wav'))


@pytest.mark.parametrize('repository', [storage.LocalOnlyFileRepository, storage.LocalFileRepository])
def test_failed_stream_preserves_previous_artifact(tmp_path, repository):
    repo = repository('job', str(tmp_path)) if repository is storage.LocalOnlyFileRepository else repository('job', str(tmp_path), Mock())
    artifact = repo.get_file('audio.wav')
    Path(artifact.file_path).write_bytes(b'previous complete audio')
    stream = Mock()
    stream.read.side_effect = [b'partial audio', OSError('stream interrupted')]
    with pytest.raises(OSError, match='stream interrupted'):
        repo.save_file_from_stream(artifact, stream)
    assert Path(artifact.file_path).read_bytes() == b'previous complete audio'
    assert list(Path(repo.directory).iterdir()) == [Path(artifact.file_path)]


def test_handle_cache_does_not_materialize_missing_source(repo):
    repo.get_file('source.mp4')
    with pytest.raises((ValueError, FileNotFoundError)):
        repo.materialize_file(RemoteFile(name='source.mp4'))
    assert not (Path(repo.directory) / 'source.mp4').exists()


def test_save_rejects_external_file_path(repo, tmp_path):
    source = tmp_path / 'outside.txt'
    source.write_text('private')
    with pytest.raises(ValueError):
        repo.save_file(RemoteFile(name='inside.txt', file_path=str(source)))


class DownloadResponse:
    def __init__(self, fail=False):
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b'first'
        if self.fail:
            raise requests.ConnectionError('connection interrupted')
        yield b'last'


def test_interrupted_download_keeps_previous_artifact(tmp_path, monkeypatch):
    destination = tmp_path / 'source.mp4'
    destination.write_bytes(b'previous complete file')
    monkeypatch.setattr(storage.requests, 'get', lambda *a, **k: DownloadResponse(fail=True))
    with pytest.raises(requests.ConnectionError):
        storage._download_file('https://example.invalid/input', str(destination))
    assert destination.read_bytes() == b'previous complete file'
    assert list(tmp_path.iterdir()) == [destination]


def test_handle_cache_downloads_and_uses_timeouts(tmp_path, monkeypatch):
    repo = storage.RemoteFileRepository('job', str(tmp_path), Mock())
    repo.get_file('source.mp4')
    called = Mock(return_value=DownloadResponse())
    monkeypatch.setattr(storage.requests, 'get', called)
    result = repo.materialize_file(RemoteFile(name='source.mp4', s3_url='https://example.invalid/input'))
    assert Path(result.file_path).read_bytes() == b'firstlast'
    assert called.call_args.kwargs['timeout'] == (10, 120)


def test_successful_retry_does_not_raise_previous_error(tmp_path, monkeypatch):
    repo = storage.LocalFileRepository('job', str(tmp_path), Mock())
    calls = []

    def download(url, path):
        calls.append(path)
        if len(calls) == 1:
            raise requests.ConnectionError('first attempt')
        Path(path).write_bytes(b'complete')

    monkeypatch.setattr(storage, '_download_file', download)
    monkeypatch.setattr('time.sleep', lambda _: None)
    result = repo.materialize_file(RemoteFile(name='source.mp4', s3_url='https://example.invalid/input'))
    assert len(calls) == 2
    assert Path(result.file_path).read_bytes() == b'complete'


@pytest.mark.parametrize('key', ['data/job/../outside', 'data/job/../../outside', 'data/job2/source.mp4', 'data/job/C:\\outside'])
def test_s3_download_rejects_traversal_and_prefix_siblings(tmp_path, key):
    client = Mock()
    client.get_paginator.return_value.paginate.return_value = [{'Contents': [{'Key': key}]}]
    with pytest.raises(ValueError):
        storage.download_s3_directory(client, 'bucket', 'data/job', str(tmp_path))
    client.download_file.assert_not_called()


def test_s3_download_creates_nested_artifact_atomically(tmp_path):
    client = Mock()
    client.get_paginator.return_value.paginate.return_value = [{'Contents': [{'Key': 'data/job/sub/file.wav'}]}]
    client.download_file.side_effect = lambda bucket, key, path: Path(path).write_bytes(b'wav')
    storage.download_s3_directory(client, 'bucket', 'data/job', str(tmp_path))
    client.get_paginator.return_value.paginate.assert_called_once_with(Bucket='bucket', Prefix='data/job/')
    assert (tmp_path / 'sub/file.wav').read_bytes() == b'wav'


def test_stream_upload_is_scoped_to_job(tmp_path):
    client = Mock()
    repo = storage.RemoteFileRepository('job', str(tmp_path), client)
    repo.save_file_from_stream(repo.get_file('audio.wav'), io.BytesIO(b'wav'))
    assert client.upload_fileobj.call_args.args[2] == str(tmp_path / 'job/audio.wav')


def test_symlink_escape_is_rejected(tmp_path):
    repo = storage.LocalOnlyFileRepository('job', str(tmp_path))
    outside = tmp_path / 'outside'
    outside.mkdir()
    try:
        (Path(repo.directory) / 'escape').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('symlink creation not permitted on this platform')
    with pytest.raises(ValueError):
        repo.get_file('escape/private.txt')
