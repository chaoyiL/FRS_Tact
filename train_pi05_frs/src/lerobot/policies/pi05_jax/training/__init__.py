"""pi0.5 training stack, vendored from openpi's `src/openpi/training/`.

Mirrors upstream module-for-module (`sharding`, `utils`, `optimizer`, `weight_loaders`,
`checkpoints`, `data_loader`, `config`) so that diffing against openpi stays cheap. Only
`data_loader` and `config` deviate, and only where they must: upstream reads datasets through the
official `lerobot` package, which collides by name with this repo's own `lerobot` package -- see
../README.md.
"""
