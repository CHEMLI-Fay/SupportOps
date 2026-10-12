"""Run reproducible AgentOps evaluations and save local JSON reports."""

import argparse
from pathlib import Path

from support_ops.agent_evaluation import report_passed, run_suite
from support_ops.config import find_project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Call the real Groq agent.')
    parser.add_argument('--judge', action='store_true', help='Score answers with DeepEval G-Eval on Groq.')
    parser.add_argument('--langsmith', action='store_true', help='Create a private LangSmith dataset/experiment.')
    parser.add_argument('--case', action='append', dest='case_ids', help='Select a case; repeat to select several.')
    parser.add_argument('--delay', type=float, default=15, help='Seconds between cases and judge calls.')
    parser.add_argument('--output', type=Path, help='Optional JSON report path.')
    args = parser.parse_args()
    if (args.judge or args.langsmith or args.case_ids) and not args.live:
        parser.error('--judge, --langsmith and --case require --live')
    if args.delay < 0:
        parser.error('--delay must be nonnegative')
    root = find_project_root(Path(__file__).parent)
    report, path = run_suite(
        root, live=args.live, judge_enabled=args.judge, publish=args.langsmith,
        case_ids=args.case_ids, delay_seconds=args.delay, output=args.output,
    )
    print(f'Report: {path}')
    print('Baseline: {passed}/{total}'.format(passed=report['offline'].get('passed', 0), total=report['offline'].get('total', 0)))
    if report.get('experiment_url'):
        print('Experiment:', report['experiment_url'])
    if report.get('infrastructure_error'):
        print('Infrastructure error:', report['infrastructure_error'])
    raise SystemExit(0 if report_passed(report) else 1)


if __name__ == '__main__':
    main()
