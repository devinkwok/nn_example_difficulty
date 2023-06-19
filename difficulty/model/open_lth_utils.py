from pathlib import Path

import sys
sys.path.append("open_lth")
from open_lth.api import iterations_per_epoch, get_hparams_dict, get_dataset_hparams


# def get_pruning_sparsity(pruning_dir: Path):
#     sparsity_file = pruning_dir / "main" / "sparsity_report.json"
#     if not sparsity_file.exists():
#         raise RuntimeError(f"Pruning sparsity file missing: {sparsity_file}")
#     with open(sparsity_file, 'r') as f:
#         sparsity_dict = json.loads(f.read())
#     return sparsity_dict["unpruned"] / sparsity_dict["total"]


# def get_iterative_magnitude_pruning_checkpoints(hparams, ckpt_dir: Path):
#     pruning_fraction = float(hparams["Pruning"]["pruning_fraction"])
#     # last_ckpt has same name as pruning ckpts
#     _, last_ckpt = get_latest_training_checkpoint(hparams, ckpt_dir)
#     ckpts = []
#     for subdir in ckpt_dir.glob("*"):
#         if subdir.is_dir() and subdir.name.startswith("level_"):
#             try:
#                 prune_level = int(subdir.name.split("_")[1])
#             except ValueError:  # not a valid checkpoint
#                 print(f"{subdir} does not contain a pruning checkpoint")
#                 continue
#             ckpt_file = subdir / "main" / last_ckpt.name
#             if not ckpt_file.exists():
#                 print(f"{ckpt_file} is not a pruning checkpoint")
#                 continue
#             # check that fractions are close
#             fraction = (1 - pruning_fraction)**prune_level
#             actual_fraction = get_pruning_sparsity(subdir)
#             if abs(fraction - actual_fraction) > 1e-4:
#                 print(f"WARNING: pruning fractions differ: calculated {fraction}, actual {actual_fraction}")
#             ckpts.append((actual_fraction, ckpt_file))
#     sorted(ckpts, key=lambda x: x[0])  # order by step
#     return list(zip(*ckpts))  # return as separate lists (steps, filenames)
