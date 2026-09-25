"""One-off runner for the real Phase 4 five-batch experiment.

Not a reusable CLI (no argparse) — `sawti.eval.experiments.grounded_graph`,
its closest sibling, has none either; both are invoked directly for a
single real run rather than repeatedly with varying flags. Kept as a
checked-in file rather than an ad-hoc `python -c` so the exact invocation
that produced eval_results.md's Phase 4 entry is reproducible from source
control, not just from a shell history.

`checkpoint_path` is set: a real run already lost 3 already-completed
batches once to a provider's daily quota exhausted mid-batch-4 (see
docs/09-DECISIONS.md). Re-running this script after an interruption resumes
from the first incomplete batch instead of starting over; a fully
successful run removes its own checkpoint file.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sawti.eval.experiments.five_batch import run_five_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

CHECKPOINT_PATH = Path("data/five_batch_checkpoint.json")

if __name__ == "__main__":
    result = run_five_batch(checkpoint_path=CHECKPOINT_PATH)
    print(f"Done. Corrections captured per batch: {result.corrections_captured}")
    print(f"Active rules after each batch: {result.active_rules_after_batch}")
