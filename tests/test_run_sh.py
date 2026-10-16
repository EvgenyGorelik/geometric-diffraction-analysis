"""Exercise launcher boundaries without requiring a Docker daemon or GPU."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.log = self.folder/'calls.jsonl'
        self.fake = self.folder/'docker'
        self.fake.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as fp:
    fp.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1:3] == ['image', 'inspect']:
    sys.exit(int(os.environ.get('IMAGE_MISSING', 0)))
if sys.argv[1] == 'build':
    sys.exit(int(os.environ.get('BUILD_EXIT', 0)))
sys.exit(int(os.environ.get('COMMAND_EXIT', 0)))
''')
        self.fake.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{self.folder}:{os.environ["PATH"]}',
                        CALL_LOG=str(self.log), GDA_GPUS='none', GDA_IMAGE='test-gda:local')
        self.env.pop('GDA_IN_CONTAINER', None)

    def run_launcher(self, *args, **env):
        return subprocess.run(['bash', str(ROOT/'run.sh'), *args], cwd=self.folder,
                              env=dict(self.env, **env), capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_auto_build_mounts_and_quoted_arguments(self):
        result = self.run_launcher('--mount', '/data/my experiment:/data:ro', 'simulate',
                                   '--hkl-file', '/data/my file.hkl', IMAGE_MISSING='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        inspect, build, run = self.calls()
        self.assertEqual(inspect, ['image', 'inspect', 'test-gda:local'])
        self.assertEqual(build, ['build', '-f', str(ROOT/'docker/Dockerfile'), '-t', 'test-gda:local', str(ROOT)])
        self.assertIn(f'{ROOT}:/workspace', run)
        self.assertIn('/data/my experiment:/data:ro', run)
        self.assertEqual(run[-4:], ['test-gda:local', 'simulate', '--hkl-file', '/data/my file.hkl'])
        self.assertNotIn('--gpus', run)

    def test_training_gpu_and_exit_status(self):
        result = self.run_launcher('--gpus', 'all', 'train', '--root_dir', 'data/my set', COMMAND_EXIT='7')
        self.assertEqual(result.returncode, 7)
        run = self.calls()[-1]
        self.assertEqual(run[run.index('--gpus')+1], 'all')
        self.assertEqual(run[-4:], ['test-gda:local', 'train', '--root_dir', 'data/my set'])

    def test_build_failure_stops_run(self):
        result = self.run_launcher('simulate', IMAGE_MISSING='1', BUILD_EXIT='8')
        self.assertEqual(result.returncode, 8)
        self.assertEqual(len(self.calls()), 2)

    def test_local_training_dispatch_and_container_simulation_dispatch(self):
        # An interpreter stand-in lets us check module handoff without torch.
        interpreter = self.folder/'python-recorder'
        interpreter.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as fp:
    fp.write(json.dumps([os.getcwd(), *sys.argv[1:]])+'\\n')
sys.exit(6)
''')
        interpreter.chmod(0o755)
        result = self.run_launcher('--local', 'train', '--root_dir', 'a directory', PYTHON=str(interpreter))
        self.assertEqual(result.returncode, 6)
        self.assertEqual(self.calls()[-1], [str(ROOT), '-m', 'training.train', '--root_dir', 'a directory'])
        result = self.run_launcher('simulate', '--config=custom.json', '--output', 'my output',
                                   GDA_IN_CONTAINER='1', GDA_PROJECT_DIR=str(ROOT), PYTHON=str(interpreter))
        self.assertEqual(result.returncode, 6)
        self.assertEqual(self.calls()[-1], [str(ROOT), '-m', 'simulation.rotation_experiment',
                                           '--config=custom.json', '--output', 'my output'])

    def test_repository_layout(self):
        visible_files = {p.name for p in ROOT.iterdir() if p.is_file() and not p.name.startswith('.')}
        self.assertEqual(visible_files, {'run.sh'})
        self.assertTrue((ROOT/'docker/Dockerfile').is_file())
        self.assertTrue((ROOT/'docker/Dockerfile.dockerignore').is_file())
        for module in ('simulation.simulate_experiment', 'training.train',
                       'training.train_segmentation', 'inference.run_inference',
                       'inference.infer_segmentation'):
            self.assertIsNotNone(importlib.util.find_spec(module), module)

    def test_unknown_command_does_not_build(self):
        result = self.run_launcher('unknown')
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.log.exists())


if __name__ == '__main__':
    unittest.main()
