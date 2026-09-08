import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import configuration
from analysis.olms_vector_table import OLMSVectorTable
from experiment_data import metadata_path, read_responses
from SLM.essay_feedback import EssayFeedback
from vector_generation import generate_vectors
from test_analysis import vector_fixture


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.essays = self.root / "essays"
        self.essays.mkdir()
        for name in ["first.md", "second.md"]:
            (self.essays / name).write_text(f"Essay {name}.", encoding="utf-8")
        self.responses = self.root / "responses.csv"
        self.vectors = self.root / "vectors.csv"
        self.model = Mock()
        self.model.run_inference.side_effect = lambda prompt, seed: f"Feedback, with punctuation.\nSeed {seed}"
        self.feedback = EssayFeedback(llm=self.model)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def collect(self, resume=False, repeats=3):
        return self.feedback.generate_multiple_feedback(
            output_path=self.responses, essays_path=self.essays, repeats=repeats, resume=resume)

    @staticmethod
    def fake_score(first, second):
        return dict(zip(OLMSVectorTable.COMPONENTS, [0.8, 0.7, 0.9, 0.6]))

    def test_collection_counts_seeds_and_resume_with_new_essay(self):
        rows = self.collect()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({call.kwargs["seed"] for call in self.model.run_inference.call_args_list}), 12)
        self.assertEqual(len(read_responses(self.responses)), 12)
        self.collect(resume=True)
        self.assertEqual(self.model.run_inference.call_count, 12)
        (self.essays / "third.md").write_text("New essay.", encoding="utf-8")
        self.assertEqual(len(self.collect(resume=True)), 18)
        self.assertEqual(self.model.run_inference.call_count, 18)
        self.assertTrue(json.loads(metadata_path(self.responses).read_text())["complete"])
        with self.assertRaises(FileExistsError):
            self.collect()
        with patch.object(configuration, "ai_recipient_information_2", "Changed prompt"):
            with self.assertRaisesRegex(ValueError, "settings changed"):
                self.collect(resume=True)

    def test_response_interruption_preserves_completed_rows(self):
        self.model.run_inference.side_effect = ["first response", RuntimeError("interrupted")]
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertEqual(len(read_responses(self.responses, complete=False)), 1)
        with self.assertRaisesRegex(ValueError, "collection is incomplete"):
            generate_vectors(self.responses, self.vectors, score_pair=self.fake_score)
        self.model.run_inference.side_effect = lambda prompt, seed: f"Resumed {seed}"
        self.assertEqual(len(self.collect(resume=True)), 12)
        self.assertEqual(read_responses(self.responses)[0]["response"], "first response")

    def test_vector_counts_resume_and_input_changes(self):
        self.collect()
        scorer = Mock(side_effect=self.fake_score)
        generate_vectors(self.responses, self.vectors, score_pair=scorer)
        table = OLMSVectorTable(self.vectors)
        self.assertEqual(scorer.call_count, 42)
        self.assertEqual(len(table.select("first.md", "AA")), 3)
        self.assertEqual(len(table.select("first.md", "AB")), 9)
        generate_vectors(self.responses, self.vectors, score_pair=scorer, resume=True)
        self.assertEqual(scorer.call_count, 42)
        with self.assertRaises(FileExistsError):
            generate_vectors(self.responses, self.vectors, score_pair=scorer)
        content = self.responses.read_text().replace("Feedback", "Changed")
        self.responses.write_text(content)
        with self.assertRaisesRegex(ValueError, "data or vector settings changed"):
            generate_vectors(self.responses, self.vectors, score_pair=scorer, resume=True)

    def test_vector_interruption_resumes_only_missing_pairs(self):
        self.collect()
        scorer = Mock(side_effect=[self.fake_score(None, None), RuntimeError("interrupted")])
        with self.assertRaises(RuntimeError):
            generate_vectors(self.responses, self.vectors, score_pair=scorer)
        self.assertEqual(len(pd.read_csv(self.vectors)), 1)
        scorer = Mock(side_effect=self.fake_score)
        generate_vectors(self.responses, self.vectors, score_pair=scorer, resume=True)
        self.assertEqual(scorer.call_count, 41)
        OLMSVectorTable(self.vectors)

    def test_legacy_and_incomplete_input_rejected(self):
        self.responses.write_text("essay_file,response_index,response\nx.md,1,Feedback\n")
        with self.assertRaisesRegex(ValueError, "Legacy"):
            read_responses(self.responses)
        self.responses.unlink()
        self.collect()
        frame = pd.read_csv(self.responses)
        frame.iloc[:-1].to_csv(self.responses, index=False)
        with self.assertRaisesRegex(ValueError, "Incomplete responses"):
            generate_vectors(self.responses, self.vectors, score_pair=self.fake_score)

    def test_cli_analysis_without_model_imports(self):
        vector_fixture(essay_count=6).to_csv(self.vectors, index=False)
        output = self.root / "analysis"
        script = (
            "import sys; from main import main; main(sys.argv[1:]); "
            "assert not any(p in sys.modules for p in "
            "['llama_cpp', 'torch', 'transformers', 'bert_score', 'ufal.udpipe', 'SLM.slm'])"
        )
        command = [sys.executable, "-c", script, "analyze", "--vectors", str(self.vectors),
                   "--output", str(output), "--bootstrap-replicates", "5000", "--seed", "12"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "report.md").exists())
        self.assertEqual(len(pd.read_csv(output / "within_essay_stability.csv")), 12)
        self.assertEqual(len(pd.read_csv(output / "within_bootstrap.csv.gz")), 60000)
        self.assertEqual(json.loads((output / "metadata.json").read_text())["seed"], 12)
        self.assertEqual(json.loads((output / "hotelling_t2.json").read_text())["status"], "unavailable")
        again = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(again.returncode, 2)
        self.assertIn("not empty", again.stderr)

    def test_help_does_not_import_model_dependencies(self):
        script = "import sys; import main; main.main([]); assert 'torch' not in sys.modules and 'llama_cpp' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("responses,vectors,analyze", result.stdout)


if __name__ == "__main__":
    unittest.main()
