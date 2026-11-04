import io
from abc import ABC
from dotenv import load_dotenv

import boto3
import requests
import os.path
import shutil
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse
from langswap.pipeline_models.models import RemoteFile
from typing import List
import os
from pathlib import Path, PureWindowsPath

load_dotenv()
BUCKET = os.getenv('BUCKET', 'debug-bucket-langswap-bucket')


def _safe_path(directory, name):
    """Resolve job-relative names without escaping through traversal or symlinks."""
    if not isinstance(name, str) or not name or '\x00' in name:
        raise ValueError('A non-empty relative file name is required')
    windows = PureWindowsPath(name)
    parts = name.replace('\\', '/').split('/')
    if windows.drive or windows.root or ':' in name or '..' in parts:
        raise ValueError(f'Unsafe repository path: {name!r}')
    path = os.path.join(directory, *parts)
    root = os.path.realpath(directory)
    resolved = os.path.realpath(path)
    if resolved == root or os.path.commonpath([root, resolved]) != root:
        raise ValueError(f'Path escapes repository: {name!r}')
    return path


def _job_directory(base_directory, public_id):
    if not isinstance(public_id, str) or '/' in public_id or '\\' in public_id:
        raise ValueError('public_id must be a single relative directory name')
    return _safe_path(base_directory, public_id)


def _download_file(url, file_path):
    if not isinstance(url, str) or urlparse(url).scheme not in ('http', 'https'):
        raise ValueError('A HTTP(S) download URL is required')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temporary_path = None
    try:
        with requests.get(url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with NamedTemporaryFile(dir=os.path.dirname(file_path), delete=False) as output:
                temporary_path = output.name
                for chunk in response.iter_content(chunk_size=8192):
                    output.write(chunk)
            if os.path.getsize(temporary_path) == 0:
                raise ValueError('Downloaded file is empty')
        os.replace(temporary_path, file_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _write_stream(stream, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=os.path.dirname(file_path), delete=False) as output:
            temporary_path = output.name
            shutil.copyfileobj(stream, output, length=8192)
        os.replace(temporary_path, file_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)



def download_s3_directory(s3_client, bucket_name, s3_prefix, local_dir):


    os.makedirs(local_dir, exist_ok=True)

    paginator = s3_client.get_paginator('list_objects_v2')
    prefix = s3_prefix.rstrip('/') + '/'
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if not key.startswith(prefix):
                    if key == s3_prefix.rstrip('/'):
                        continue
                    raise ValueError(f'S3 key is outside requested prefix: {key!r}')
                relative_path = key[len(prefix):]
                if not relative_path:
                    continue
                local_path = _safe_path(local_dir, relative_path)

                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                if not key.endswith('/'):
                    temporary_path = None
                    try:
                        with NamedTemporaryFile(dir=os.path.dirname(local_path), delete=False) as output:
                            temporary_path = output.name
                        s3_client.download_file(bucket_name, key, temporary_path)
                        os.replace(temporary_path, local_path)
                    finally:
                        if temporary_path and os.path.exists(temporary_path):
                            os.unlink(temporary_path)



class FileRepository(ABC):

    def _path(self, name):
        return _safe_path(self._directory, name)

    def _saved_path(self, file):
        expected = self._path(file.name)
        if not file.file_path or os.path.realpath(file.file_path) != os.path.realpath(expected):
            raise ValueError('Saved file must belong to this repository')
        return expected

    def __init__(self, public_id, base_directory: str, s3_client: boto3.client):
        ...

    def materialize_file(self, file: RemoteFile) -> RemoteFile:
        """ Saves file from file.s3_url to file.file_path """
        ...

    def get_file(self, file_name: str) -> RemoteFile:
        """ Creates empty RemoteFile object with file.file_path = directory / file_name
         and file.name == file_name and """
        ...

    def save_file(self, file: RemoteFile, force: bool = False):
        """ Uploads file to s3 using _s3_client.
        :returns: RemoteFile with file.s3_url"""
        ...

    def save_file_from_stream(self, file: RemoteFile, stream: io.BytesIO):
        """ Saves :param: stream to file.file_path """
        ...

    def subdir(self, dir_name: str) -> str:
        """ Creates folder self.directory / dir_name.
        :returns: new folder path"""
        ...

    def save_dir(self, dir_name: str) -> List[RemoteFile]:
        """ Creates folder self.directory / dir_name.
        :returns: new folder path"""
        ...

    @property
    def directory(self) -> str:
        ...


class RemoteFileRepository(FileRepository):
    _directory: str
    _s3_client: boto3.client
    _download_chunk_size = 8192

    _cached_files: dict[str, RemoteFile]

    def __init__(self, public_id, base_directory: str, s3_client: boto3.client):
        super().__init__(public_id, base_directory, s3_client)
        self._directory = _job_directory(base_directory, public_id)
        self._s3_client = s3_client
        self._cached_files = {}
        os.makedirs(self._directory, exist_ok=True)

    def materialize_file(self, file: RemoteFile) -> RemoteFile:
        file_path = self._path(file.name)
        cached = self._cached_files.get(file.name)
        if cached and os.path.isfile(file_path) and cached.s3_url == file.s3_url:
            return cached
        _download_file(file.s3_url, file_path)
        remote_file = RemoteFile(
            name=file.name,
            file_path=file_path,
            s3_url=file.s3_url,
        )

        self._cached_files[file.name] = remote_file

        return remote_file

    def get_file(self, file_name: str):
        self._path(file_name)
        cached = self._cached_files.get(file_name)
        if cached is not None:
            return cached
        remote_file = RemoteFile(
            name=file_name,
            file_path=self._path(file_name)
        )
        self._cached_files[file_name] = remote_file
        return remote_file

    def save_file(self, file: RemoteFile, force: bool = False):
        self._saved_path(file)
        print(f"bucket: {BUCKET}, file: {file.file_path}")
        if file.s3_url and not force:
            return file

        with open(file.file_path, 'rb') as f:
            self._s3_client.upload_fileobj(io.BytesIO(f.read()), BUCKET, file.file_path)
            s3_url = self._s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': BUCKET,
                    'Key': file.file_path
                },
                ExpiresIn=60 * 60 * 48  # 48 hours
            )
        file.s3_url = s3_url
        return file

    def save_file_from_stream(self, file: RemoteFile, stream: io.BytesIO):
        file_path = self._path(file.name)
        print(f"BUCKET: {BUCKET}")
        self._s3_client.upload_fileobj(stream, BUCKET, file_path)
        s3_url = self._s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': BUCKET,
                'Key': file_path
            },
            ExpiresIn=60 * 60 * 24  # 24 hours
        )
        return RemoteFile(
            name=file.name,
            file_path=self._path(file.name),
            s3_url=s3_url,
        )

    @property
    def directory(self) -> str:
        return self._directory

    def subdir(self, dir_name: str) -> str:
        new_dir = self._path(dir_name)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    def save_dir(self, dir_name):
        self._path(os.path.relpath(dir_name, self._directory))
        # print(dir_name)
        #list_files = list(map(lambda x: str(x).replace(dir_name, ""), Path(dir_name).rglob("*.*")))
        list_files = [str(path) for path in Path(dir_name).rglob('*') if path.is_file()]
        remove_file_collection = []
        for file_name in list_files:
            # print(file_name, self._directory)
            local_file = RemoteFile(
                name=os.path.relpath(file_name, self._directory),
                file_path=file_name, #os.path.join(self._directory, file_name)
            )
            remote_file = self.save_file(local_file)
            remove_file_collection.append(remote_file)
        return list_files

class LocalOnlyFileRepository(FileRepository):
    """File repository that works entirely locally without S3."""
    _directory: str
    _cached_files: dict[str, RemoteFile]

    def __init__(self, public_id: str, base_directory: str):
        self._directory = _job_directory(base_directory, public_id)
        self._cached_files = {}
        os.makedirs(self._directory, exist_ok=True)

    def materialize_file(self, file: RemoteFile) -> RemoteFile:
        file_path = self._path(file.name)
        cached = self._cached_files.get(file.name)
        if cached and os.path.isfile(file_path) and cached.s3_url == file.s3_url:
            return cached
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # If file_path is already set and exists, use it
        if file.file_path and os.path.exists(file.file_path):
            if os.path.realpath(file.file_path) != os.path.realpath(file_path):
                shutil.copy2(file.file_path, file_path)
        elif file.s3_url:
            _download_file(file.s3_url, file_path)
        else:
            raise FileNotFoundError(f'No source available for {file.name!r}')

        remote_file = RemoteFile(
            name=file.name,
            file_path=file_path,
            s3_url=file.s3_url or Path(file_path).resolve().as_uri(),
        )
        self._cached_files[file.name] = remote_file
        return remote_file

    def get_file(self, file_name: str):
        self._path(file_name)
        cached = self._cached_files.get(file_name)
        if cached is not None:
            return cached
        remote_file = RemoteFile(
            name=file_name,
            file_path=self._path(file_name)
        )
        self._cached_files[file_name] = remote_file
        return remote_file

    def save_file(self, file: RemoteFile, force: bool = False):
        self._saved_path(file)
        # Local-only: just return the file as-is
        if not os.path.isfile(file.file_path):
            raise FileNotFoundError(file.file_path)
        if not file.s3_url:
            file.s3_url = Path(file.file_path).resolve().as_uri()
        return file

    def save_file_from_stream(self, file: RemoteFile, stream: io.BytesIO):
        file_path = self._path(file.name)
        _write_stream(stream, file_path)
        return RemoteFile(
            name=file.name,
            file_path=file_path,
            s3_url=Path(file_path).resolve().as_uri(),
        )

    @property
    def directory(self) -> str:
        return self._directory

    def subdir(self, dir_name: str) -> str:
        new_dir = self._path(dir_name)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    def save_dir(self, dir_name):
        self._path(os.path.relpath(dir_name, self._directory))
        list_files = [str(path) for path in Path(dir_name).rglob('*') if path.is_file()]
        result = []
        for file_name in list_files:
            self._path(os.path.relpath(file_name, self._directory))
            local_file = RemoteFile(
                name=os.path.relpath(file_name, self._directory),
                file_path=file_name,
                s3_url=Path(file_name).resolve().as_uri(),
            )
            result.append(local_file)
        return list_files


class LocalFileRepository(FileRepository):
    _directory: str
    _download_chunk_size = 8192
    _s3_client: boto3.client

    _cached_files: dict[str, RemoteFile]

    def __init__(self, public_id, base_directory: str, s3_client: boto3.client):
        super().__init__(public_id, base_directory, s3_client)

        self._directory = _job_directory(base_directory, public_id)
        self._s3_client = s3_client
        self._cached_files = {}
        os.makedirs(self._directory, exist_ok=True)

    def materialize_file(self, file: RemoteFile) -> RemoteFile:
        file_path = self._path(file.name)
        cached = self._cached_files.get(file.name)
        if cached and os.path.isfile(file_path) and cached.s3_url == file.s3_url:
            return cached
        if not os.path.exists(file_path):
            last_error = None
            for i in range(5):
                try:
                    _download_file(file.s3_url, file_path)
                    last_error = None
                    break
                except (requests.RequestException, OSError) as e:
                    print(e)
                    last_error = e
                    import time
                    time.sleep(5)

            if last_error is not None:
                raise last_error
        remote_file = RemoteFile(
            name=file.name,
            file_path=file_path,
            s3_url=file.s3_url,
        )

        self._cached_files[file.name] = remote_file

        return remote_file

    def get_file(self, file_name: str):
        self._path(file_name)
        cached = self._cached_files.get(file_name)
        if cached is not None:
            return cached
        remote_file = RemoteFile(
            name=file_name,
            file_path=self._path(file_name)
        )
        self._cached_files[file_name] = remote_file
        return remote_file

    def save_file(self, file: RemoteFile, force: bool = False):
        self._saved_path(file)
        if not force:
            return file

        with open(file.file_path, 'rb') as f:
            self._s3_client.upload_fileobj(io.BytesIO(f.read()), BUCKET, file.file_path)
            s3_url = self._s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': BUCKET,
                    'Key': file.file_path
                },
                ExpiresIn=60 * 60 * 14  # 14 days
            )
        file.s3_url = s3_url
        return file

    def save_file_from_stream(self, file: RemoteFile, stream: io.BytesIO):
        file_path = self._path(file.name)
        _write_stream(stream, file_path)
        return RemoteFile(
            name=file.name,
            file_path=file_path,
            s3_url=file.s3_url or Path(file_path).resolve().as_uri(),
        )

    @property
    def directory(self) -> str:
        return self._directory

    def subdir(self, dir_name: str) -> str:

        new_dir = self._path(dir_name)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    def save_dir(self, dir_name):
        self._path(os.path.relpath(dir_name, self._directory))
        #list_files = list(map(lambda x: str(x).replace(dir_name, ""), Path(dir_name).rglob("*.*")))
        #list_files = list(map(str, Path(dir_name).rglob("*.*")))
        list_files = [str(path) for path in Path(dir_name).rglob('*') if path.is_file()]
        remove_file_collection = []
        for file_name in list_files:
            #print(file_name, self._directory)
            local_file = RemoteFile(
                name=os.path.relpath(file_name, self._directory),
                file_path=file_name, #os.path.join(self._directory, file_name)
            )
            remote_file = self.save_file(local_file)
            remove_file_collection.append(remote_file)
        return list_files
