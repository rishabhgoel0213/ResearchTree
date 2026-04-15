# Example Data

This directory holds the local dataset cache for the Parameter Golf example.

Use:

```bash
cd examples/parameter_golf
pixi run download-data
```

That downloads the published FineWeb export into:

- `data/datasets/`
- `data/tokenizers/`
- `data/manifest.json`

These files are intentionally gitignored. The example scorer and MCTS configs default to this local data directory.
