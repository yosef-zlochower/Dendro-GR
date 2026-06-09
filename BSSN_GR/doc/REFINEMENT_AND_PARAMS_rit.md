# RIT refinement modes & parameters

This documents the RIT refinement features ported onto `master` in branch
`rit_devel_next` (from the legacy `chi-tests-SiS-limit` branch): Sphere-in-Sphere /
Box-in-Box static refinement, and two constraint-based refinement modes. It also
records the **`BSSN_REFINEMENT_MODE` renumbering** and the dropped features, so old
par files can be migrated.

> Implementation lives in `BSSN_GR/src/refinement_sis.cpp` /
> `BSSN_GR/include/refinement_sis.h`, dispatched from `BSSNCtx::is_remesh()`.

---

## 1. `BSSN_REFINEMENT_MODE` renumbering (migration-critical)

`master` added `BH_WAMR = 4`, and the legacy SiS+WAMR *combination* mode was dropped.
The new numbering therefore differs from the legacy branch:

| Mode | Legacy `chi-tests-SiS-limit` value | New `rit_devel_next` value |
|------|:----------------------------------:|:--------------------------:|
| `WAMR` / `EH` / `EH_WAMR` / `BH_LOC` | 0 / 1 / 2 / 3 | 0 / 1 / 2 / 3 (unchanged) |
| `BH_WAMR` | — (did not exist) | 4 |
| `SPHERE_IN_SPHERE` | 4 | **5**  ← changed |
| `SIS_OUT_WAMR_IN` | 5 | **removed** |
| `CONSTRAINT` | 6 | 6 (unchanged) |
| `CONSTRAINT_ERROR` | 7 | 7 (unchanged) |

> ⚠️ **A legacy par file with `BSSN_REFINEMENT_MODE = 4` (Sphere-in-Sphere) must be
> changed to `5`.** Modes 6 and 7 keep their numbers.

A par file written for the current `master` (modes 0–4) runs unchanged.

---

## 2. Dropped features (no longer exist)

- Refinement mode **`SIS_OUT_WAMR_IN`** (the SiS-outer / plain-WAMR-inner combination).
- Parameters **`BSSN_REFINEMENT_NUM_MODES`** and
  **`BSSN_REFINEMENT_MODE_COMBINATION_ORDER`** (only the dropped combination used them).

---

## 3. New refinement modes

### `SPHERE_IN_SPHERE` (value 5) — Sphere-in-Sphere / Box-in-Box
Static, geometry-only refinement. Each element is assigned to the nearer puncture and
refined to a target level determined by which concentric shell (`BSSN_BOX_RADII_*`) it
falls in. `BSSN_BOX_TYPE` selects spherical (Euclidean distance, `0`) or box
(L-infinity distance, non-zero) shells.

### `CONSTRAINT` (value 6) — value-based constraint refinement
Refines on the **value** of `log10|grad2_chi / chi^2|`, binned against the
`BSSN_CHI_VALUES[]` thresholds, with a majority vote over each element's nodes.

> ⚠️ **Known issue:** a suspected bug in `isRemeshConstraint` is under review
> (see the `TODO(rit)` marker in `refinement_sis.cpp`). Treat this mode's output with
> caution until verified.

### `CONSTRAINT_ERROR` (value 7) — constraint-based WAMR
Wavelet (WAMR) refinement driven by the chi constraint
`C_GRAD_GRAD2_CHI_EXPRESSION` rather than an evolution variable. Before
`BSSN_SIS_TO_CONSTRAINT_WAMR_TRANSITION_TIME` it uses Sphere-in-Sphere; after, it
switches to WAMR, with Sphere-in-Sphere limiting how far WAMR may de/refine. An inner
region (`BSSN_INNER_SIS_REGION_OUTER_BOUND`) and an outer light-cone region are
excluded from WAMR.

---

## 4. New parameters

| Parameter | Type | Default | Units / meaning |
|-----------|------|:-------:|-----------------|
| `WALL_TIME` | double | `1e300` | **MINUTES**. Terminate + checkpoint when wall-clock exceeds this. |
| `BSSN_MINDEPTH_SIS` | uint | `7` | Minimum refinement level for SiS / Box-in-Box. |
| `BSSN_BOX_TYPE` | uint | `0` | `0` = sphere (Euclidean), else box (L-infinity). |
| `BSSN_BOX_NUM_LEVELS` | uint[2] | `{0,0}` | #shells for [BH1, BH2]. **Required** for `SPHERE_IN_SPHERE` / `CONSTRAINT_ERROR`. |
| `BSSN_BOX_RADII_1` | double[] | — | Shell radii for BH1 (length `BSSN_BOX_NUM_LEVELS[0]`, ≤ 20). **Required** as above. |
| `BSSN_BOX_RADII_2` | double[] | — | Shell radii for BH2 (length `BSSN_BOX_NUM_LEVELS[1]`, ≤ 20). **Required** as above. |
| `BSSN_CHI_VALUES` | double[] | — | `log10|grad2_chi/chi^2|` thresholds for `CONSTRAINT` (length `BSSN_CHI_NUM_VALUES`, ≤ 20). |
| `BSSN_CHI_NUM_VALUES` | uint | `9` | Number of populated `BSSN_CHI_VALUES`. |
| `BSSN_SIS_TO_CONSTRAINT_WAMR_TRANSITION_TIME` | double | `60.0` | Coord time at which `CONSTRAINT_ERROR` switches SiS→WAMR. |
| `BSSN_INNER_SIS_REGION_OUTER_BOUND` | double | `5.0` | Outer radius of the inner SiS-only region for `CONSTRAINT_ERROR`. |
| `BSSN_REL_ERR_MIN` | double | `1.0` | Floor of the relative wavelet error `Δf / max(BSSN_REL_ERR_MIN, |f|)` (max-based, not additive). **Requires a dendrolib fork whose `compute_wavelets_3D` accepts it — otherwise the build fails** (see below). |
| `BSSN_RESTORE_WAVELET_TOL_FROM_CHECKPOINT` | bool | `false` | See §5. |
| `BSSN_RIT_DUMP_WAVELET_ERROR` | bool | `false` | If true, `CONSTRAINT_ERROR` dumps per-element wavelet error to VTU (diagnostic). |

Missing required `BSSN_BOX_*` parameters under `SPHERE_IN_SPHERE` / `CONSTRAINT_ERROR`
cause a clear abort in `is_remesh()`. Parameters are read from both TOML and (for
arrays) the manual block in `parameters.cpp`; new par files should prefer TOML.

---

## 5. Changed semantics / gotchas

- **`WALL_TIME` is in MINUTES** (the legacy code at one point used seconds). It accepts
  an integer or a float in the par file.
- **Wavelet tolerance on restart** is now taken from the par file, not the checkpoint
  (`BSSN_RESTORE_WAVELET_TOL_FROM_CHECKPOINT` defaults to `false`). This lets you change
  `BSSN_WAVELET_TOL` when resuming. Set it `true` to reproduce stock-master behaviour
  (lock the tolerance to its checkpointed value). The tolerance is still written into
  every checkpoint regardless.
- **`BSSN_CONSTRAINT_NUM_VARS` is now 9** (6 base + `C_GRAD_CHI`, `C_GRAD2_CHI`,
  `C_GRAD_GRAD2_CHI_EXPRESSION`). Constraint indices 0–5 are unchanged; VTU constraint
  output is still gated by `BSSN_NUM_CONST_VARS_VTU_OUTPUT`, so existing par files
  output the same fields. Index 6/7/8 select the new chi constraints.
- **`BSSN_REL_ERR_MIN`** is passed to `WaveletEl::compute_wavelets_3D` as the trailing
  `rel_min` argument, giving a denominator floor of `max(BSSN_REL_ERR_MIN, |f|)` (the
  default `1.0` reproduces stock behaviour). This call is **unconditional** — there is no
  feature-macro fallback. You **must** build against a dendrolib fork whose
  `compute_wavelets_3D` accepts the extra argument (point CMake at it via
  `DENDRO_dendrolib_DIR` or `DENDRO_dendrolib_GIT_TAG`); building against a dendrolib that
  lacks it **fails to compile at the call site, by design**, so `BSSN_REL_ERR_MIN` can
  never be silently ignored.

---

## 6. Example par snippets (TOML)

Sphere-in-Sphere (mode 5):
```toml
BSSN_REFINEMENT_MODE = 5
BSSN_BOX_TYPE        = 0          # 0 = sphere, else box
BSSN_MINDEPTH_SIS    = 7
BSSN_BOX_NUM_LEVELS  = [6, 6]
BSSN_BOX_RADII_1     = [220.0, 110.0, 55.0, 25.0, 10.0, 5.0]
BSSN_BOX_RADII_2     = [220.0, 110.0, 55.0, 25.0, 10.0, 5.0]
```

Constraint-based WAMR (mode 7):
```toml
BSSN_REFINEMENT_MODE                         = 7
BSSN_BOX_NUM_LEVELS                          = [6, 6]
BSSN_BOX_RADII_1                             = [220.0, 110.0, 55.0, 25.0, 10.0, 5.0]
BSSN_BOX_RADII_2                             = [220.0, 110.0, 55.0, 25.0, 10.0, 5.0]
BSSN_SIS_TO_CONSTRAINT_WAMR_TRANSITION_TIME  = 60.0
BSSN_INNER_SIS_REGION_OUTER_BOUND            = 5.0
WALL_TIME                                    = 1380   # minutes
```
