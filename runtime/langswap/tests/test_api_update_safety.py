"""Remote update persistence and UI credential isolation without external calls."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from langswap.file_repository import LocalOnlyFileRepository, RemoteFileRepository
from langswap.pipeline_models.models import TranslationPipelineConfig, save_config_to_json


@pytest.fixture
def update_env(tmp_path, monkeypatch):
    import langswap.api as api
    repo = LocalOnlyFileRepository('job', str(tmp_path))
    events = []
    saved = []
    captured = []
    options = {'restored_source': True, 'public_id': 'job'}

    def restore(client, bucket, prefix, directory):
        events.append('restore')
        assert bucket == api.BUCKET
        assert directory == repo.directory
        config = TranslationPipelineConfig(
            source_lang='english', target_lang='french', public_id=options['public_id'],
            source_video_path='/retired/runtime/source.mp4', base_dir='/retired/runtime',
            tts_model='edge', translation_backend='openai', speaker_voices={'SPEAKER_00': 'fr-FR-DeniseNeural'},
        )
        save_config_to_json(config, str(Path(repo.directory) / 'config.json'))
        if options['restored_source']:
            (Path(repo.directory) / 'source.mp4').write_bytes(b'restored source')

    def materialize(repository, url):
        assert events == ['restore']
        events.append('materialize')
        handle = repository.get_file('source.mp4')
        Path(handle.file_path).write_bytes(b'current supplied source')
        return handle.file_path

    def save(handle, force=False):
        saved.append((handle.name, force, Path(handle.file_path).read_bytes()))
        handle.s3_url = f'https://storage.invalid/job/{handle.name}'
        return handle

    class Pipeline:
        def __init__(self, config, file_repository):
            captured.append(config)
            self.config = config
            self.video_translation = SimpleNamespace(processed_video=SimpleNamespace(s3_url='https://storage.invalid/result'))

        def translate_video(self):
            return self.video_translation

        def generate_srt_files(self):
            handles = []
            for name in ['source_transcript.srt', 'translated_transcript.srt']:
                handle = repo.get_file(name)
                handle.s3_url = 'https://storage.invalid/old-subtitles'
                Path(handle.file_path).write_bytes(b'new subtitles')
                handles.append(handle)
            return handles

    class Changes:
        def __init__(self, pipeline, translation):
            self.translation = translation

        def apply_update_translations(self, updates):
            assert updates[0].text == 'nouvelle traduction'
            (Path(repo.directory) / 'translations.json').write_text(json.dumps({'translation': updates[0].text}))
            (Path(repo.directory) / 'translations.cache.json').write_text('{"fingerprint":"same-input"}')
            generated = Path(repo.subdir('generated_audio'))
            (generated / 'segment.wav').write_bytes(b'new audio')
            (generated / 'segment.wav.cache.json').write_text('{"text":"new"}')
            return self.translation

    monkeypatch.setattr(api, 'BASE_DIR', str(tmp_path))
    monkeypatch.setattr(api, 'init_s3_client', Mock())
    monkeypatch.setattr(api, 'RemoteFileRepository', lambda *args: repo)
    monkeypatch.setattr(api, 'download_s3_directory', restore)
    monkeypatch.setattr(api, 'get_file', materialize)
    monkeypatch.setattr(api, 'VideoTranslationPipeline', Pipeline)
    monkeypatch.setattr(api, 'ChangeManager', Changes)
    monkeypatch.setattr(repo, 'save_file', Mock(side_effect=save))
    monkeypatch.setattr(repo, 'save_dir', Mock(wraps=repo.save_dir))
    return SimpleNamespace(api=api, repo=repo, saved=saved, captured=captured, events=events, options=options)


@pytest.mark.parametrize('supply_source', [False, True])
def test_update_restores_then_materializes_and_republishes_changes(update_env, supply_source):
    env = update_env
    request = {'public_id': 'job', 'update_translation': [[0, 'nouvelle traduction']]}
    if supply_source:
        request['s3_video_url'] = 'https://input.invalid/new.mp4'
    result = env.api.process_update_translation(request)
    config = env.captured[0]
    assert config.source_video_path == str(Path(env.repo.directory) / 'source.mp4')
    assert config.base_dir == env.api.BASE_DIR
    assert config.source_lang == 'english'
    assert config.target_lang == 'french'
    assert config.tts_model == 'edge'
    assert config.translation_backend == 'openai'
    assert config.speaker_voices == {'SPEAKER_00': 'fr-FR-DeniseNeural'}
    expected_source = b'current supplied source' if supply_source else b'restored source'
    assert Path(config.source_video_path).read_bytes() == expected_source
    for name in ['translations.json', 'translations.cache.json', 'source_transcript.srt', 'translated_transcript.srt']:
        assert any(saved_name == name and force for saved_name, force, data in env.saved)
    for name in ['source_transcript.srt', 'translated_transcript.srt']:
        assert next(data for saved_name, force, data in env.saved if saved_name == name) == b'new subtitles'
    if supply_source:
        assert ('source.mp4', True, b'current supplied source') in env.saved
    env.repo.save_dir.assert_called_once()
    assert result['s3_translated_transcript_url'].endswith('/translated_transcript.srt')


def test_update_rejects_configuration_from_another_job(update_env):
    update_env.options['public_id'] = 'other-job'
    with pytest.raises(ValueError, match='different job'):
        update_env.api.process_update_translation({'public_id': 'job', 'update_translation': []})
    assert not update_env.captured


def test_update_requires_source_when_old_snapshot_has_none(update_env):
    update_env.options['restored_source'] = False
    with pytest.raises(FileNotFoundError, match='supply s3_video_url'):
        update_env.api.process_update_translation({'public_id': 'job', 'update_translation': []})
    assert not update_env.captured


def test_directory_upload_includes_sidecars_and_skips_nested_directories(tmp_path):
    client = Mock()
    repo = RemoteFileRepository('job', str(tmp_path), client)
    generated = Path(repo.subdir('generated_audio'))
    references = generated / '.speaker_refs'
    references.mkdir()
    (references / 'reference.wav').write_bytes(b'reference')
    (generated / 'segment.wav').write_bytes(b'audio')
    (generated / 'segment.wav.cache.json').write_bytes(b'cache')
    (generated / 'extensionless').write_bytes(b'metadata')
    repo.save_dir(str(generated))
    keys = {call.args[2] for call in client.upload_fileobj.call_args_list}
    assert keys == {str(path) for path in generated.rglob('*') if path.is_file()}


def test_initial_remote_job_persists_source_snapshot(update_env, monkeypatch):
    env = update_env
    source = env.repo.get_file('source.mp4')
    Path(source.file_path).write_bytes(b'initial source')
    monkeypatch.setattr(env.api, 'get_file', lambda *args: source.file_path)
    pipeline = Mock()
    pipeline.config.dubbing_algo = 'speedup'
    pipeline._merge.return_value.processed_video.s3_url = 'https://storage.invalid/result'
    pipeline.generate_srt_files.return_value = [SimpleNamespace(s3_url='https://storage.invalid/source.srt'),
                                                 SimpleNamespace(s3_url='https://storage.invalid/french.srt')]
    monkeypatch.setattr(env.api, 'VideoTranslationPipeline', Mock(return_value=pipeline))
    env.api.process_translation({'public_id': 'job', 'target_language': 'french', 'tts_engine': 'openvoice',
                                 's3_video_url': 'https://input.invalid/source'})
    assert ('source.mp4', True, b'initial source') in env.saved


def test_gradio_rejects_request_token_without_mutating_operator_credentials(monkeypatch):
    import gradio_demo
    monkeypatch.setenv('ELEVEN_API_KEY', 'operator-key')
    result = gradio_demo.translate_video(None, 'english', 'french', 'elevenlabs', 'vad', 'marian',
                                         'speedup', 'cpu', False, 'request-key')
    assert result[:3] == (None, None, None)
    assert 'operator .env' in result[3]
    assert 'request-key' not in result[3]
    assert gradio_demo.os.environ['ELEVEN_API_KEY'] == 'operator-key'
    next_result = gradio_demo.translate_video(None, 'english', 'french', 'elevenlabs', 'vad', 'marian',
                                              'speedup', 'cpu', False)
    assert 'No video' in next_result[3]
    assert gradio_demo.os.environ['ELEVEN_API_KEY'] == 'operator-key'


def test_gradio_form_defaults_and_absence_of_request_credential_field():
    import gradio_demo
    with gradio_demo.gr.Blocks() as demo:
        gradio_demo.build_translate_tab()
    components = demo.get_config_file()['components']
    props = {component['props'].get('label'): component['props'] for component in components}
    assert props['Langue source']['value'] == 'english'
    assert props['Langue de sortie']['value'] == 'french'
    assert props['Dossier de sortie']['value']
    assert props['Version à produire']['value'] == 'local'
    assert props['Skip speaker diarization (use only for a single speaker)']['value'] is False
    assert not any('token' in str(label).lower() for label in props)
