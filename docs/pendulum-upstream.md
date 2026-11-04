# Official NeuralKoopmanSVD comparator on the pendulum

The optional adapter in [pendulum_upstream.py](../experiments/pendulum_upstream.py) imports the authors' actual implementation from [MinchanJeong/NeuralKoopmanSVD at d76194d](https://github.com/MinchanJeong/NeuralKoopmanSVD/tree/d76194d5b7e62d973edb2a226f2a54f4662b48cb). This is a bounded adaptation to the pendulum task, not new measurements on their original paper datasets or a reproduction of their long training campaigns. No upstream source is vendored. The upstream code is [MIT licensed, © 2025 Minchan Jeong and Jongha Jon Ryu](https://github.com/MinchanJeong/NeuralKoopmanSVD/blob/d76194d5b7e62d973edb2a226f2a54f4662b48cb/LICENSE).

## Which components actually run?

- Two separate authors' `MLPEncoder`s with `CenteringWrapper`, and their `KoopmanPLModule` with `NestedLoRALoss(nesting="jnt", reg_weight=0)` and its Adam optimizer. The total rank includes an explicit constant. As in the authors' default configuration, singular values are not separately parameterized (`learn_svals=False`).
- Unlike our independent marginal minibatch objective, their loss computes both Gram terms on the same minibatch of paired transitions. The harness preserves this distinction. We use one CPU device and a short explicit optimization loop rather than the authors' distributed Lightning Trainer.
- Native inference **upstream_cca** and **upstream_edmd** invokes their `KoopmanEstimator.partial_fit(...).finalize()` on all training transitions and their `predict` for each horizon (aligned CCA and raw forward EDMD(f), respectively). It decodes the scalar cosine endpoint. This native estimator uses one pooled, time-homogeneous operator and only uses the length of the future damping schedule; it does not select different transition operators from that schedule.
- **upstream_prefix** uses the identical trained upstream encoders with *our* damping-prefix-weighted alignment, cross-time bridges and readout. This third row isolates feature differences at a common schedule-aware head; the resulting predictor is **not** native upstream inference. All three rows share a single network fit per view/seed/rank.
- Our benchmark harness specifies a width-16 tanh network on each side, ranks 4/8 including the constant, train-only input scaling, 64 Adam updates (learning rate 0.003, batch 128), gradient norm clip 10, and validation nested-loss checkpoints at zero and every eighth update. These choices are matched to the local bounded study, not taken from the authors' larger benchmark configurations. There is no tuning or calibration on test transitions. The supplied scripts do not report GPU/distributed speed or general accuracy of the full authors' software.

## Reproducing the optional arm

From a project environment with its usual CPU PyTorch requirements installed (Python 3.12 verified):

```bash
git clone https://github.com/MinchanJeong/NeuralKoopmanSVD.git ../pendulum-upstream
git -C ../pendulum-upstream checkout d76194d5b7e62d973edb2a226f2a54f4662b48cb
python -m pip install 'lightning==2.6.6' 'torchmetrics==1.9.0' scipy
export KOOPMAN_UPSTREAM="$(cd ../pendulum-upstream && pwd)"
python -m pytest -q tests/test_pendulum_upstream.py
```

Run the main [pendulum study](../experiments/pendulum_study.py) with its official-comparison option and the same checkout environment. The adapter does not clone, install, or download anything when imported. It checks the upstream commit, source-tree cleanliness, and import location. The native estimator only needs NumPy/SciPy, PyTorch and Lightning/torchmetrics for this experiment: SchNetPack, MD trajectory data, wandb and a GPU are unnecessary. Installing numeric dependencies in your own environment is permitted; preserve all upstream license notices in any redistributed checkout.

[Jeong, Ryu, Yun and Wornell, NeurIPS 2025](https://arxiv.org/abs/2507.07222) report the training principle and inference approaches. Exact executed source paths: [loss](https://github.com/MinchanJeong/NeuralKoopmanSVD/blob/d76194d5b7e62d973edb2a226f2a54f4662b48cb/koopmansvd/models/losses.py), [module](https://github.com/MinchanJeong/NeuralKoopmanSVD/blob/d76194d5b7e62d973edb2a226f2a54f4662b48cb/koopmansvd/models/lightning.py), [inference](https://github.com/MinchanJeong/NeuralKoopmanSVD/blob/d76194d5b7e62d973edb2a226f2a54f4662b48cb/koopmansvd/models/inference/estimator.py).
