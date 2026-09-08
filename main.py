"""Independent terminal stages for the OLMS stability pilot."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiment_data import file_digest, metadata_path


def collect_responses(args):
  from SLM.essay_feedback import EssayFeedback

  rows = EssayFeedback().generate_multiple_feedback(
    output_path=args.output, essays_path=args.essays,
    repeats=args.repeats, resume=args.resume
  )

  print(f"Saved {len(rows)} responses to {args.output}")


def calculate_vectors(args):
  from vector_generation import generate_vectors

  output = generate_vectors(args.responses, args.output, resume=args.resume,
                            parser_dir=args.parser_dir, diagnostics=args.diagnostics)
  print(f"Saved OLMS vectors to {output}")


def analyze_vectors(args):
  from analysis.analysis_writer import AnalysisWriter
  from analysis.between_essay_stability_analyzer import BetweenEssayStabilityAnalyzer
  from analysis.bootstrap_confidence_interval import BootstrapConfidenceInterval
  from analysis.hotelling_t2_test import HotellingT2Test
  from analysis.olms_vector_table import OLMSVectorTable
  from analysis.prompt_shift_analyzer import PromptShiftAnalyzer
  from analysis.response_level_bootstrapper import ResponseLevelBootstrapper
  from analysis.within_essay_stability_analyzer import WithinEssayStabilityAnalyzer

  # Load the OLMSVector CSV and validate
  table = OLMSVectorTable(args.vectors)
  source_metadata = None
  if metadata_path(args.vectors).exists():
    source_metadata = json.loads(metadata_path(args.vectors).read_text(encoding="utf-8"))
    if not source_metadata.get("complete"):
      raise ValueError("Vector generation is incomplete. Resume the vectors command first.")
    if set(table.essays) != set(source_metadata["essay_files"]):
      raise ValueError("Vector CSV does not contain all essays recorded in its metadata.")
    if table.response_count != source_metadata["response_count"]:
      raise ValueError("Vector response count differs from its metadata.")

  # Create helper objects
  intervals = BootstrapConfidenceInterval(args.confidence)
  bootstrapper = ResponseLevelBootstrapper(args.bootstrap_replicates, args.seed)
  writer = AnalysisWriter(args.output)

  
  within, distances = WithinEssayStabilityAnalyzer().analyze(table)
  between = BetweenEssayStabilityAnalyzer().analyze(within)
  shifts, overall = PromptShiftAnalyzer().analyze(table)
  print(f"Bootstrapping {len(table.essays)} essays, {args.bootstrap_replicates} replicates each...", flush=True)
  bootstrap = bootstrapper.run(table)
  within_metrics = ["mean_distance"] + [f"{c}_mean" for c in table.COMPONENTS]
  shift_components = [f"delta_{c}" for c in table.COMPONENTS]
  shift_metrics = ["magnitude"] + shift_components
  tables = {
    "within_essay_stability": within,
    "within_essay_intervals": intervals.table(
      bootstrap["within_bootstrap"], ["essay_file", "prompt"], within_metrics),
    "between_essay_stability": between,
    "prompt_shift": shifts,
    "prompt_shift_intervals": intervals.table(
      bootstrap["prompt_shift_bootstrap"], ["essay_file"], shift_metrics),
    "overall_prompt_shift": overall,
    "overall_prompt_shift_intervals": intervals.table(
      bootstrap["overall_prompt_shift_bootstrap"], [], shift_metrics, shift_components),
  }
  hotelling = HotellingT2Test().run(shifts[shift_components].to_numpy(dtype=float))
  metadata = {
      "created_at": datetime.now(timezone.utc).isoformat(),
      "vector_file": str(Path(args.vectors).resolve()), "vector_sha256": file_digest(args.vectors),
      "essay_count": len(table.essays), "essay_files": table.essays,
      "response_count": table.response_count, "components": table.COMPONENTS,
      "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed,
      "confidence": args.confidence, "bootstrap_unit": "response within essay and prompt",
      "essay_resampling": False, "component_scaling": "none", "distance": "euclidean",
      "sd_ddof": 1, "source_metadata": source_metadata,
  }
  for name, frame in tables.items():
      writer.write_table(name, frame)
  writer.write_table("vector_distances", distances)
  for name, frame in bootstrap.items():
      writer.write_table(name, frame, compressed=True)
  writer.write_json("hotelling_t2", hotelling)
  writer.write_json("metadata", metadata)
  report = writer.write_report(tables, hotelling, metadata)
  print(f"Saved analysis to {args.output}; report: {report}")


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
  parser = argparse.ArgumentParser(description="Run one stage of the OLMS stability pilot.")
  commands = parser.add_subparsers(dest="command")

  responses = commands.add_parser("responses", help="Collect and save repeated A/B responses.")
  responses.add_argument("--essays", type=Path, help="Essay directory (defaults to configuration.essays_path).")
  responses.add_argument("--repeats", type=positive_integer, default=10, help="Responses per essay per prompt (default: 10; minimum: 2).")
  responses.add_argument("--output", type=Path, default=Path("data/feedback_data.csv"))
  responses.add_argument("--resume", action="store_true", help="Generate only missing responses in a compatible dataset.")
  responses.set_defaults(run=collect_responses)

  vectors = commands.add_parser("vectors", help="Compute saved-response AA, BB, AB, and self-comparison vectors.")
  vectors.add_argument("--responses", type=Path, default=Path("data/feedback_data.csv"))
  vectors.add_argument("--output", type=Path, default=Path("data/feedback_data_olms_vectors.csv"))
  vectors.add_argument("--parser-dir", type=Path, default=Path(".models/udpipe"))
  vectors.add_argument("--resume", action="store_true", help="Compute only missing pairs in a compatible checkpoint.")
  vectors.add_argument("--diagnostics", action="store_true", help="Append sentence and structure diagnostics alongside vectors.")
  vectors.set_defaults(run=calculate_vectors)

  analysis = commands.add_parser("analyze", help="Analyze saved vectors without loading language or scoring models.")
  analysis.add_argument("--vectors", type=Path, default=Path("data/feedback_data_olms_vectors.csv"))
  analysis.add_argument("--output", type=Path, default=Path("data/analysis"))
  analysis.add_argument("--bootstrap-replicates", type=positive_integer, default=5000)
  analysis.add_argument("--seed", type=int, default=42, help="Nonnegative bootstrap random seed (default: 42).")
  analysis.add_argument("--confidence", type=float, default=0.95)
  analysis.set_defaults(run=analyze_vectors)
  
  return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    try:
        args.run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()


# python main.py responses --output data/pilot_responses.csv
# python main.py vectors --responses data/pilot_responses.csv --output data/pilot_vectors.csv
# python main.py analyze --vectors data/pilot_vectors.csv --output data/pilot_analysis