import json
from pathlib import Path


def notebook_source():
    notebook = json.loads(Path('notebook/kaggle.ipynb').read_text())
    return '\n'.join(''.join(cell['source']) if isinstance(cell['source'], list) else cell['source'] for cell in notebook['cells'])


def test_kaggle_notebook_has_bounded_smoke_before_all_pipeline_comparison():
    source = notebook_source()
    assert '--pipeline all --data oto-speech --size_gb 1 --max-samples 1 --max-seconds 10' in source
    assert source.rindex('--pipeline all --data oto-speech --size_gb 1') > source.index('--max-seconds 10')
    assert 'HUGGINGFACE_TOKEN' in source
    assert 'hf auth login --token' not in source


def test_kaggle_notebook_uses_uploaded_zip_not_git_clone():
    source = notebook_source()
    assert 'duplexchat_project.zip' in source
    assert '/kaggle/working/duplexchat_project' in source
    assert 'git clone' not in source
    assert 'git pull' not in source
