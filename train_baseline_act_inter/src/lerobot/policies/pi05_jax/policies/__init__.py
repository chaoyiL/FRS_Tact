"""Dataset-specific input/output transforms, mirroring openpi's `src/openpi/policies/`.

Upstream ships one module per robot platform (`aloha_policy.py`, `droid_policy.py`,
`libero_policy.py`). This repo trains on the bimanual pick_tube datasets, so it ships
`pick_tube_policy.py`, written against the same `transforms.DataTransformFn` contract.
"""
