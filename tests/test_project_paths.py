"""Path portability checks; no network or private datasets required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProjectPathsTest(unittest.TestCase):
    def probe(self, code, cwd, overrides=None):
        env = {k: v for k, v in os.environ.items()
               if k not in {"TEAM_DIR", "PROJECT_DIR", "STEP5_DIR", "S08_ROOT"}}
        env.update(overrides or {})
        result = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                                text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_script_defaults_do_not_depend_on_working_directory(self):
        code = ("import json,runpy; p=runpy.run_path(" + repr(str(ROOT / "scripts/project_paths.py"))
                + "); print(json.dumps({k:str(p[k]) for k in "
                + "['REPO_ROOT','DATA_DIR','PROJECT_DIR','STEP5_DIR']}))")
        with tempfile.TemporaryDirectory() as outside:
            for cwd in [ROOT, ROOT / "scripts", outside]:
                paths = self.probe(code, cwd)
                self.assertEqual(paths["REPO_ROOT"], str(ROOT))
                self.assertEqual(paths["DATA_DIR"], str(ROOT / "files csv"))
                self.assertEqual(paths["PROJECT_DIR"], str(ROOT / "3_4_surprise_returns"))
                self.assertEqual(paths["STEP5_DIR"], str(ROOT / "5_data cleaning"))

    def test_external_data_keeps_code_in_checkout(self):
        code = ("import json,runpy; p=runpy.run_path(" + repr(str(ROOT / "5_data cleaning/code/settings.py"))
                + "); print(json.dumps({k:str(p[k]) for k in "
                + "['NOV2_DATA','CODE','NOV2_CODE','PART34','STEP5_DIR']}))")
        with tempfile.TemporaryDirectory() as data:
            paths = self.probe(code, data, {"TEAM_DIR": data})
            self.assertEqual(paths["NOV2_DATA"], str(Path(data).resolve() / "files csv"))
            self.assertEqual(paths["CODE"], str(ROOT / "5_data cleaning/code"))
            self.assertEqual(paths["NOV2_CODE"], str(ROOT / "1_2nov" / "code"))
            self.assertEqual(paths["PART34"], str(Path(data).resolve() / "3_4_surprise_returns"))
            self.assertEqual(paths["STEP5_DIR"], str(Path(data).resolve() / "5_data cleaning"))

    def test_all_notebook_bootstraps(self):
        with tempfile.TemporaryDirectory() as outside:
            for notebook in ROOT.rglob("*.ipynb"):
                cells = json.loads(notebook.read_text())["cells"]
                setup = next("".join(c["source"]).split("# Notebook paths.")[0]
                             for c in cells if "# Notebook paths." in "".join(c["source"]))
                code = setup + '\nimport json; print(json.dumps([str(REPO_ROOT),str(NOV2_DATA)]))'
                for cwd, env in [(ROOT, {}), (notebook.parent, {}),
                                 (outside, {"S08_ROOT": str(ROOT)})]:
                    with self.subTest(notebook=notebook.name, cwd=cwd):
                        self.assertEqual(self.probe(code, cwd, env),
                                         [str(ROOT), str(ROOT / "files csv")])

    def test_no_colab_paths_or_relative_python_commands(self):
        for notebook in ROOT.rglob("*.ipynb"):
            for cell in json.loads(notebook.read_text())["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                for obsolete in ["/content/", "google.colab", "!python ", "!pip ", "os.chdir("]:
                    self.assertNotIn(obsolete, source, str(notebook))


if __name__ == "__main__":
    unittest.main()
