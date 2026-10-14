"""The tool folder can be copied intact, with one separate directory per run."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tifffile


SCRIPTS = Path(__file__).resolve().parents[1]


class PortableLayoutTests(unittest.TestCase):
    def test_utf8_reports_with_windows_style_default_encoding(self):
        """Report generation must not depend on the Windows ANSI code page."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # Emulate a non-UTF-8 default for Path text I/O on every test host.
            # Explicit encoding arguments must still take precedence.
            (root / 'sitecustomize.py').write_text(
                'from pathlib import Path\n'
                'original_open = Path.open\n'
                'def legacy_open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):\n'
                '    if "b" not in mode and encoding is None:\n'
                '        encoding = "cp1252"\n'
                '    return original_open(self, mode, buffering, encoding, errors, newline)\n'
                'Path.open = legacy_open\n', encoding='utf-8')
            inputs = root / 'Raw images Δ'
            inputs.mkdir()
            yy, xx = np.mgrid[:128, :128]
            nucleus = 10 + 500 * ((yy - 64)**2 + (xx - 64)**2 < 26**2)
            arr = np.array([nucleus, np.full((128, 128), 20), np.full((128, 128), 20)], dtype=np.uint16)
            image = inputs / 'contrôle_Δ.TIF'
            tifffile.imwrite(image, arr, imagej=True, resolution=(5, 5),
                             metadata={'unit': 'micron', 'axes': 'CYX'})
            original_image = image.read_bytes()
            parameters = json.loads((SCRIPTS / 'parameters.template.json').read_text(encoding='utf-8'))
            parameters['group_names'] = ['contrôle_Δ']
            parameters['channels']['2'] = 'γH2AX'
            config = root / 'parameters Δ.json'
            config.write_text(json.dumps(parameters, ensure_ascii=False), encoding='utf-8')
            output = root / 'ORION results Δ'
            env = dict(os.environ, PYTHONPATH=str(root), PYTHONUTF8='0',
                       PYTHONIOENCODING='utf-8', MPLCONFIGDIR=str(root / 'cache'))
            for script, args in [
                ('analyze_foci.py', ['--input-dir', inputs, '--parameters', config,
                                     '--output-dir', output, '--foci-channels', '2', '--qc-output', 'minimal']),
                ('validate_results.py', ['--output-dir', output]),
            ]:
                result = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                                        cwd=root, env=env, capture_output=True, text=True, encoding='utf-8')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            review = (output / 'Review.html').read_text(encoding='utf-8')
            self.assertIn('contrôle_Δ', review)
            self.assertIn('γH2AX', review)
            methods = (output / 'Methods/Methods.md').read_text(encoding='utf-8')
            self.assertIn('γH2AX', methods)
            self.assertTrue((output / 'Methods/Validation_Report.json').is_file())
            self.assertEqual(image.read_bytes(), original_image)

    def test_flat_copy_creates_independent_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            for path in SCRIPTS.glob('*.py'):
                shutil.copy2(path, project / path.name)
            shutil.copy2(SCRIPTS / 'parameters.template.json', project / 'experiment.json')
            input_dir = project / 'raw'
            input_dir.mkdir()
            yy, xx = np.mgrid[:128, :128]
            nucleus = 10 + 500 * ((yy - 64)**2 + (xx - 64)**2 < 26**2)
            arr = np.array([nucleus, np.full((128, 128), 20), np.full((128, 128), 20)], dtype=np.uint16)
            tifffile.imwrite(input_dir / 'test.tif', arr, imagej=True, resolution=(5, 5),
                             metadata={'unit': 'micron', 'axes': 'CYX'})
            original_parameters = (project / 'experiment.json').read_bytes()
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(project / 'cache'))

            def run(script, *args, success=True):
                result = subprocess.run([sys.executable, str(project / script), *map(str, args)],
                                        cwd=project, env=env, capture_output=True, text=True)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            for _ in range(2):
                result = run('analyze_foci.py', '--input-dir', 'raw', '--parameters', 'experiment.json', '--foci-channels', 2)
                self.assertIn('Analysis output:', result.stdout)
            runs = list((project / 'Analyses').iterdir())
            self.assertEqual(len(runs), 2)
            for out in runs:
                self.assertEqual((out / 'Methods/parameters.json').read_bytes(), original_parameters)
                effective = json.loads((out / 'Methods/Effective_Parameters.json').read_text(encoding='utf-8'))
                self.assertEqual(effective['foci_channels'], [2])
                self.assertTrue((out / 'Results/Cell_Summary.csv').exists())
                manifest = json.loads((out / 'Methods/Run_Manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(manifest['output_directory'], str(out.resolve()))
                run('validate_results.py', '--output-dir', out)
            self.assertEqual((project / 'experiment.json').read_bytes(), original_parameters)
            self.assertFalse((project / 'Results').exists())
            self.assertFalse((project / 'Methods').exists())
            self.assertFalse((project / 'Masks').exists())
            run('make_outputs.py', '--help')
            for script in ['make_outputs.py', 'validate_results.py']:
                result = run(script, success=False)
                self.assertIn('--output-dir', result.stderr)
            # Automatic output generation uses the sibling script and the new run
            # folder, even when launched from another working directory. A failed
            # renderer must preserve completed analysis and report failure.
            (project / 'make_outputs.py').write_text(
                'import argparse\nfrom pathlib import Path\n'
                'parser=argparse.ArgumentParser()\n'
                'parser.add_argument("--output-dir",type=Path,required=True)\n'
                'parser.add_argument("--qc-output",choices=["minimal"],required=True)\n'
                'args=parser.parse_args()\n'
                'assert (args.output_dir/"Results/Image_Summary.csv").exists()\n'
                'raise SystemExit(7)\n')
            result=run('analyze_foci.py','--input-dir','raw','--parameters','experiment.json',
                       '--foci-channels',2,'--qc-output','minimal',success=False)
            self.assertEqual(result.returncode,7)
            self.assertIn('Output generation failed; completed analysis is saved',result.stderr)
            self.assertEqual(len(list((project/'Analyses').iterdir())),3)


if __name__ == '__main__':
    unittest.main()
