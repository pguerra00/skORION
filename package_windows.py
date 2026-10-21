#!/usr/bin/env python3
"""Build a portable source ZIP without local environments or analysis data."""
import argparse
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'dist' / 'scikitORION-windows.zip'


def build_bundle(output):
    files = set()
    for pattern in ('*.py', '*.md', '*.bat', '*.command', '*.mjs'):
        files.update(ROOT.glob(pattern))
    files.update(ROOT / name for name in (
        'LICENSE', 'requirements.txt', 'parameters.template.json',
        '.gitattributes', '.gitignore', '.github/workflows/windows.yml',
    ))
    files.update((ROOT / 'tests').glob('test*.py'))
    files.update(path for path in (ROOT / 'assets').rglob('*')
                 if path.is_file() and not path.name.startswith('.'))
    # Read each required file before replacing a previously usable bundle.
    contents = [(path.relative_to(ROOT).as_posix(), path.read_bytes())
                for path in sorted(files)]
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='orion-package-', dir=output.parent) as folder:
        temporary = Path(folder) / output.name
        with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
            for name, data in contents:
                # Command Prompt scripts retain Windows line endings in ZIPs
                # built from either a Mac checkout or a Windows checkout.
                if name.endswith('.bat'):
                    data = data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
                archive.writestr('scikitORION/' + name, data)
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP integrity verification failed')
        temporary.replace(output)
    return output, len(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='Destination ZIP (default: dist/scikitORION-windows.zip)')
    args = parser.parse_args()
    output, count = build_bundle(args.output)
    print(f'Created {output} ({count} files)')


if __name__ == '__main__':
    main()
