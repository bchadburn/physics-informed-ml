# Experiment 2: FNO vs UNet for 2D Darcy Flow

Solves the steady-state heat equation −∇·(κ∇T) = 1 with T = 0 on the boundary,
learning the solution operator κ → T using a Fourier Neural Operator (FNO2d) and
a UNet2d baseline.

## Running

**Training:**
```bash
uv run python experiments/exp2_fno_heat_transfer/train.py
```

**Evaluation (benchmark table):**
```bash
uv run python experiments/exp2_fno_heat_transfer/evaluate.py
# Evaluate UNet2d instead:
uv run python experiments/exp2_fno_heat_transfer/evaluate.py model.name=unet2d
```

## FNO Resolution Invariance

The Fourier Neural Operator operates in frequency space: it lifts inputs to a
feature space, applies learned multiplications to the lowest-frequency Fourier
coefficients, then projects back. Because the spectral modes being learned
(e.g. modes1=12, modes2=12) represent global low-frequency patterns that exist
at any spatial resolution, the same trained weights can be applied to grids of
different sizes without any retraining — as long as the grid is at least
2×modes in each dimension. This zero-shot resolution generalization is a
fundamental advantage over convolution-based models like UNet2d, whose filters
encode spatial scale implicitly and do not transfer across resolutions.

## Results

See `results.md` after running `evaluate.py`.
