"""E3 step-size ablation: tail-averaged error of the CE (scaled) critic for three base step sizes.

Usage (from the exp/ folder):  python3 e3_ablation.py .
"""
import os
import sys

import numpy as np

HERE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], HERE]

# Reuse the setup and the run() function from e3_sgd.py without running its main loop.
src = open(os.path.join(HERE, "e3_sgd.py")).read().split("out = {")[0]
exec(src)

for lr in [0.3, 0.1, 0.03]:
    e, Q = run("ce-scaled", 0, outer=400, lr=lr)
    print(lr, "tail-avg median rel err", float(np.median(np.abs(Q - pred) / pred)))
