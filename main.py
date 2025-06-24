from evaluator.eval_cls_vid import generate_comprehensive_report
from extractor.ucf_video import eval
import argparse

def main():
    parser = argparse.ArgumentParser(description="Video classification evaluation and reporting script.")
    parser.add_argument(
        "--mode",
        choices=["eval", "report"],
        required=True,
        help="Select 'eval' to run evaluation or 'report' to generate report."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/results.csv",
        help="Path to the results CSV file for reporting (used with --mode report)."
    )

    args = parser.parse_args()

    if args.mode == "eval":
        eval()
    elif args.mode == "report":
        _ , _ = generate_comprehensive_report(args.csv)

if __name__ == "__main__":
    main()