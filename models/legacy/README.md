# Legacy models (INCOMPATIBLE — do not deploy)

These are the original model zips, moved out of `models/` because their
observation spaces do not match the app's feature engine and never did:

| File | Obs space | Actions | Notes |
|------|-----------|---------|-------|
| `hunter_apex_1500_brain.zip`    | `Box(-inf, inf, (5,), float32)`  | `Discrete(2)` | SB3 2.8.0 / NumPy 2.0.2, ~5.2M steps |
| `guardian_apex_1500_brain.zip`  | `Box(-inf, inf, (5,), float32)`  | `Discrete(2)` | SB3 2.8.0 / NumPy 2.0.2, ~5.15M steps |
| `executive_apex_manager.zip`    | `Box(-inf, inf, (3,), float32)`  | `Discrete(3)` | ~1M steps |

The app's `feature_engine.py` produces 15-dim market features (17-dim for
Guardian/Executive), so every `predict()` call on these models raised a shape
error that was previously swallowed. The training notebooks that defined the
5-dim/3-dim features are permanently lost, making these zips untrainable and
unusable.

They are kept only for the historical record. Retrain replacements with
`training/train_triad.py` (see `training/README.md`); `app/ai_brains.py` now
refuses to load any model whose observation shape doesn't match the feature
engine, so these can never silently trade again.
