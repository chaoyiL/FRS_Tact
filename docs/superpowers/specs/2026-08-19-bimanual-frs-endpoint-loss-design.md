# Bimanual FRS Composite-Endpoint Loss Design

## Goal

Train one tactile-conditioned FRS velocity field for a 20-dimensional bimanual action chunk while preventing an inactive hand from drifting away from the frozen SmolVLA action when only the other hand has meaningful tactile contact.

The approved design gives every sample one coherent joint endpoint:

- the left 10 action dimensions independently choose between GT and VLA using the left-wrist gate;
- the right 10 action dimensions independently choose between GT and VLA using the right-wrist gate;
- one flow-matching trajectory runs from the cached `x_base` to that joint endpoint;
- decode, low-safety, rank, and repair remain present and become hand-aware.

This change does not add gate values to the decoder input. The decoder must infer the steering regime from the tactile history and state it already receives, preserving decoder input version 2 and the current deployment interface.

## Fixed Data Contract

This design applies to the current VB3/pick-tube 20D action contract:

| Slice | Meaning |
| --- | --- |
| `0:3` | left translation |
| `3:9` | left rotation-6D |
| `9` | left gripper |
| `10:13` | right translation |
| `13:19` | right rotation-6D |
| `19` | right gripper |

The implementation must validate `action_dim == 20` when bimanual composite-endpoint loss is enabled. It must not infer the split as `action_dim // 2` for arbitrary datasets.

The tactile names describe the two sensor faces on each wrist, not the robot hand. Gate grouping is:

- left wrist: `tactile_left_0`, `tactile_right_0`;
- right wrist: `tactile_left_1`, `tactile_right_1`.

## Per-Hand Gates

For wrist `h` with two tactile tokens, compute change against the corresponding episode-first-frame tokens:

\[
s_h=\frac12\sum_{k\in h}\left(1-\cos(e^{current}_k,e^{baseline}_k)\right).
\]

Map change to a raw gate with the existing sigmoid calibration:

\[
w_h=\sigma\left(\frac{s_h-\tau}{T}\right).
\]

Use the existing three-region remapping independently for each wrist:

\[
g_h=\operatorname{clip}\left(\frac{w_h-l}{u-l},0,1\right),
\qquad l=0.3,\ u=0.7.
\]

The raw global gate remains available for backward-compatible reporting only:

\[
w=\frac{w_L+w_R}{2}.
\]

Neither `w_h` nor `g_h` receives gradients. The initial implementation retains the same `tau` and temperature for both wrists, exposes both per-hand distributions in training and validation metrics, and treats recalibration as a later data-driven change.

## Composite Endpoint and Flow Matching

Let `G` be the GT action chunk and `P` the frozen SmolVLA action chunk. Define the single per-sample endpoint `Y` by action slice:

\[
Y_L=g_LG_L+(1-g_L)P_L,
\]

\[
Y_R=g_RG_R+(1-g_R)P_R,
\]

\[
Y=\operatorname{concat}(Y_L,Y_R).
\]

The FRS trains on one coherent path rather than separate full-action GT and VLA paths:

\[
x_t=(1-t)x_{base}+tY,
\]

\[
v^*=Y-x_{base},
\]

\[
L_{FM}=\operatorname{mean}_{H,D}\left(v_\theta(x_t,t,c)-v^*\right)^2.
\]

This replaces both existing gated `gt_fm` and `vla_fm` computations. History should report the new value as `train_loss_composite_fm`; the old two columns may remain as explicit zero-valued compatibility fields for existing plotting readers, but must not misleadingly divide the new loss between them. `train_loss_total` and `train_flow_loss` retain their current meanings.

`gate_lambda` no longer scales a VLA FM branch because no separate VLA branch exists. The bimanual mode rejects the `gate_lambda` field rather than silently pretending it remains active; the shipped bimanual YAML removes it.

## Decode and Auxiliary Losses

Decode once from `x_base` with the configured differentiable solver, exactly as deployment will decode:

\[
\hat A=\operatorname{Decode}_\theta(x_{base},c).
\]

Define per-hand MSE values by averaging over horizon and the 10 dimensions of that hand:

\[
d^G_h=MSE_h(\hat A,G),\quad
d^P_h=MSE_h(\hat A,P),\quad
b_h=MSE_h(P,G).
\]

Thresholded hand-group auxiliary reductions (low-safety, rank, and repair) use active-group normalization. Each wrist is normalized independently, then the active wrist scalars are averaged. An empty active group contributes zero and is excluded from the active-wrist denominator, so a batch containing high-gate samples for only one wrist does not halve that auxiliary loss. Direct decode is defined for both wrists on every sample and uses an ordinary two-wrist mean.

### Direct decode

Retain direct decode supervision for both selected endpoints:

\[
L_{decode}=\lambda_{decode}\frac12\sum_{h\in\{L,R\}}
\left[g_hd^G_h+(1-g_h)d^P_h\right].
\]

Unlike the old high-gate-only term, this directly anchors an inactive wrist to the VLA endpoint. A single decoded tensor is reused for every auxiliary term.

### Low-gate safety

Retain the existing nearest-endpoint safety hinge per wrist using raw gates:

\[
L_{low,h}=\lambda_{low}\operatorname{WMean}_{(1-w_h)\mathbf1[w_h\le l]}
\left[\operatorname{ReLU}(\min(d^G_h,d^P_h)-\delta_{low})\right].
\]

This term is partly redundant with inactive-wrist decode-to-VLA supervision, but it remains for the requested first experiment and for direct comparison with existing ablations.

### Rank

Retain rank but make it wrist-local so one wrist cannot hide the other wrist's failure:

\[
L_{rank,h}=\lambda_{rank}\operatorname{WMean}_{w_h\mathbf1[w_h\ge u]}
\left[\operatorname{ReLU}(d^G_h-d^P_h+m_{rank})\right].
\]

The configured balanced-mean and worst-source-CVaR aggregation modes continue to apply, now over wrist-local penalties. For CVaR, a source/wrist pair is an independent active group.

### Repair

Retain repair as a wrist-local comparison against the frozen VLA baseline:

\[
L_{repair,h}=\lambda_{repair}\operatorname{WMean}_{w_h\mathbf1[w_h\ge u]}
\left[\operatorname{ReLU}(d^G_h-b_h+m_{repair})\right].
\]

The current YAML weight is zero, so the term remains implemented and logged but contributes no gradient until configured otherwise.

## Total Objective

The bimanual gated objective is:

\[
L=L_{FM}+L_{decode}+L_{low}+L_{rank}+L_{repair}.
\]

Here `low`, `rank`, and `repair` are the active-wrist means of their left/right terms defined above.

The ordinary batch reduction and optional dataset-balanced reduction retain their current semantics. MSE values continue to use normalized model-space actions and include every horizon step.

## Training and Deployment Behavior

The FRS parameter tree and decoder input remain unchanged. Existing decoder-input-v2 deployment code can load a newly trained checkpoint after normal checkpoint metadata validation. Deployment does not need `w_L` or `w_R` to execute the network.

The loss is a soft training guarantee, not a hard runtime guarantee, because both wrists share model parameters. The existing inactive-arm XYZ runtime protection remains an independent last-resort safeguard. A future per-hand residual blend at deployment is explicitly outside this change.

Compute changes are favorable relative to the current gated objective:

- flow matching drops from two velocity evaluations (full GT and full VLA) to one composite-endpoint evaluation;
- FireFlow decode still runs once and is reused by decode, safety, rank, and repair;
- no extra tactile ResNet pass is introduced.

## Configuration and Checkpoint Metadata

Add an explicit bimanual composite-endpoint loss mode rather than silently changing old checkpoint semantics. The run configuration and checkpoint metadata must record:

- loss mode/version;
- left and right action slices;
- left and right tactile-token groups;
- gate thresholds, tau, and temperature;
- auxiliary weights and aggregation mode.

Resume must reject checkpoints with the old scalar-gate objective or a different action/tactile grouping. Old checkpoints remain loadable for evaluation and deployment through their existing decoder configuration.

## Metrics and Evaluation

Training history and validation output must add:

- `gate_w_left`, `gate_w_right`, and their quantiles;
- left/right MSE to GT and VLA;
- left/right frozen-VLA baseline MSE to GT;
- left/right low-safety violation fraction;
- left/right high-gate rank satisfaction and GT gain;
- composite-endpoint FM and decode terms.

Checkpoint selection must not allow one wrist's gain to cancel the other wrist's regression. Feasibility checks are evaluated per wrist, and the selection key uses the worse wrist before aggregate error.

## Error Handling

Training must fail early when:

- bimanual mode is selected with an action dimension other than 20;
- configured action slices overlap, leave gaps, or exceed the action dimension;
- tactile groups do not resolve to exactly two tokens per wrist;
- a resumed checkpoint uses incompatible loss metadata;
- any gate, endpoint, loss, or decoded action contains non-finite values.

## Test Strategy

Unit tests must cover:

1. wrist token grouping uses `_0` for left wrist and `_1` for right wrist;
2. `w_L=1,w_R=0` produces endpoint `[GT_L,VLA_R]`;
3. `w_L=0,w_R=1` produces endpoint `[VLA_L,GT_R]`;
4. both-zero and both-one gates reduce exactly to full VLA and full GT endpoints;
5. soft gates interpolate only their own 10D slice;
6. composite FM performs one model velocity call and targets `Y-x_base`;
7. decode, safety, rank, and repair cannot mix errors across wrists;
8. empty active wrist groups return finite zero losses;
9. source-balanced and worst-source-CVaR reductions preserve wrist separation;
10. old scalar-gate checkpoints cannot resume bimanual training;
11. training history and evaluation metrics contain the new per-wrist fields;
12. decoder parameter structure and deployment inference remain backward compatible.

An integration test should use the single-contact case to verify that one optimization step sends the active wrist toward GT and the inactive wrist toward VLA. A short loss ablation should compare FM-only against FM plus decode to quantify endpoint drift and training cost before changing decode frequency or step count.
