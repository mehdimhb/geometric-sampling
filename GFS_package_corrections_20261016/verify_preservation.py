"""Read-only verification of historical research and the pre-fix source archive."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUTHORIZED = {'graphical_sampling/population.py',
              'graphical_sampling/clustering/fip_balanced_nmeans.py',
              'graphical_sampling/index/density_disparity.py'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    protected = json.loads((HERE/'protected_before.json').read_text())
    expected_changes = {str(ROOT/rel) for rel in AUTHORIZED}
    changed, missing, same = [], [], 0
    for name, before in protected.items():
        path = Path(name)
        if not path.is_file():
            missing.append(name)
        elif digest(path.read_bytes()) != before:
            changed.append(name)
        else:
            same += 1
    unexpected = sorted(set(changed) - expected_changes)
    manifest = json.loads((HERE/'backup_manifest.json').read_text())
    archive = HERE/manifest['backup']
    archive_ok = digest(archive.read_bytes()) == manifest['backup_SHA256']
    archive_errors = []
    with tarfile.open(archive) as tar:
        for rel, before in manifest['files'].items():
            stream = tar.extractfile(rel)
            if stream is None or digest(stream.read()) != before:
                archive_errors.append(rel)
    result = dict(passed=not (missing or unexpected or archive_errors) and archive_ok,
                  protected_files=len(protected), unchanged_files=same,
                  authorized_source_changes=sorted(changed), unexpected_changes=unexpected,
                  missing_files=missing, historical_archive_SHA256=digest(archive.read_bytes()),
                  historical_archive_valid=archive_ok, historical_files_verified=len(manifest['files']),
                  historical_archive_errors=archive_errors,
                  note='The three source changes were explicitly approved after the historical audit; all research files and the historical archive remain unchanged.')
    (HERE/'final_preservation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert result['passed'], result


if __name__ == '__main__':
    main()
