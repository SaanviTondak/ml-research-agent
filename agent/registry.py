"""Building a task from command-line arguments.

Kept out of loop.py so the loop stays a search algorithm and knows nothing
about which benchmarks exist. Adding a task means adding a line here and a
package under tasks/, not touching the agent.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASKS = ("kuairand", "tabular", "synthetic")
DEFAULT_TASK = "kuairand"


def add_arguments(ap):
    ap.add_argument("--task", default=DEFAULT_TASK, choices=TASKS,
                    help="which benchmark to run against")
    g = ap.add_argument_group(
        "tabular task",
        "for --task tabular: a directory of train.csv / valid.csv / test.csv")
    g.add_argument("--data", default=None,
                   help="directory holding your CSVs")
    g.add_argument("--target", default=None,
                   help="name of the column to predict")
    g.add_argument("--metric", default="rmse",
                   help="rmse | mae | logloss | auc | accuracy")
    g.add_argument("--holdout", default="test",
                   help="split whose target column is removed (default: test)")
    return ap


def build(args):
    """Instantiate the task named on the command line, and prepare its data."""
    name = getattr(args, "task", DEFAULT_TASK)

    if name == "kuairand":
        from tasks.kuairand.task import KuaiRandTask
        return KuaiRandTask()

    if name == "synthetic":
        from tasks.synthetic.task import SyntheticTask
        task = SyntheticTask()
        task.prepare(force=False, verbose=True)
        return task

    if name == "tabular":
        missing = [f"--{f}" for f in ("data", "target")
                   if not getattr(args, f, None)]
        if missing:
            raise SystemExit(
                f"--task tabular needs {' and '.join(missing)}.\n"
                f"  example: python3 -m agent.loop --task tabular \\\n"
                f"             --data ./mydata --target churn --metric auc")
        src = Path(args.data)
        if not src.is_dir():
            raise SystemExit(f"--data {src} is not a directory")
        for required in ("train.csv", "valid.csv"):
            if not (src / required).exists():
                raise SystemExit(f"{src / required} is required and missing")

        # Validate here rather than letting a constructor or a firewall
        # raise: everything below is a mistake a first-time user makes, and a
        # traceback is a worse answer than a sentence.
        import csv as _csv
        from tasks.tabular import evaluate as ev
        if args.metric not in ev.DIRECTION:
            raise SystemExit(
                f"unknown metric {args.metric!r}; "
                f"choose from {', '.join(sorted(ev.DIRECTION))}")
        with open(src / "train.csv", newline="") as fh:
            columns = next(_csv.reader(fh), [])
        if args.target not in columns:
            raise SystemExit(
                f"target column {args.target!r} is not in "
                f"{src / 'train.csv'}.\n"
                f"  columns are: {', '.join(columns)}")

        from tasks.tabular.task import TabularTask
        task = TabularTask(source=src, target=args.target,
                           metric=args.metric, holdout=args.holdout,
                           name=f"tabular:{src.name}:{args.metric}")
        print(f"preparing {task.name} ...")
        task.prepare(force=True, verbose=True)
        return task

    raise SystemExit(f"unknown task {name!r}; have {', '.join(TASKS)}")
