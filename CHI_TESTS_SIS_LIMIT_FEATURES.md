# Features of branch `chi-tests-SiS-limit` (for reimplementation on top of `master`)

**Merge-base with master:** `c63392e5fc207f38c400dc392d22fedbb8414df1`
**Goal:** reimplement the *features* below on top of the redesigned `master`, with the
simplest possible changes. Git history does not need to be preserved.

Diff surveyed: `git diff c63392e…414df1 chi-tests-SiS-limit -- BSSN_GR/include BSSN_GR/src`
(12 files, ~1423 insertions). Plus the `dendrolib` submodule (separate section below).

---

## 0. Reimplementation gotchas (read first)

- **`dendrolib` is a git submodule** and the branch pins a *different, dirty* commit than
  master. Some features (`OCT_IGNORE`, get-refinement-flags-by-eleid) live there, not in the
  parent repo. See §7.
- **`RefinementMode` enum collision.** Master is now
  `{ WAMR=0, EH, EH_WAMR, BH_LOC, BH_WAMR }` — i.e. master added `BH_WAMR=4`.
  The branch instead added `SPHERE_IN_SPHERE=4, SIS_OUT_WAMR_IN=5, CONSTRAINT=6,
  CONSTRAINT_ERROR=7`. When porting, **append** the new modes after master's `BH_WAMR`,
  do not reuse value 4. (`grDef.h:97`)
- **`VAR_CONSTRAINT` enum size.** Master has 6 constraint vars (`C_HAM … C_PSI4_IMG`).
  The branch appends 3 more and bumps `BSSN_CONSTRAINT_NUM_VARS` 6→9. Append at the end.
- Much of the parent-repo diff in `bssngr_main.cpp` is **removal of the AEH apparent-horizon
  solver** and debug prints — *not* features. Don't reintroduce those (see §6).

---

## 1. Verification of the recalled list

| Recalled item | Verdict | Notes |
|---|---|---|
| Main: remesh called differently at init | **Already in master** | Mechanism is `BSSN_USE_SET_REF_MODE_FOR_INITIAL_CONVERGE` in `BSSNCtx::init_grid()` (`bssnCtx.cpp:383`). It exists at the merge-base **and** in current master — the branch did **not** change it. Nothing to reimplement unless master's redesign dropped it (it didn't). |
| Main: walltime calc, terminate-on-walltime, periodic checkpoints | **Confirmed** | See §2. WALL_TIME is in **minutes**. |
| Main: constraints calc changed for regridding | **Confirmed** | New `BSSNCtx::calc_constraints()` wrapper, called before remesh. See §3. |
| Constraints: hand-added new variables (best done as small edits) | **Confirmed** | `grad_chi`, `grad2_chi`, `grad_grad2_chi_expression`. 3 small edits each in `grDef.h`, `physcon.cpp`, `physconeqs.cpp`. See §4. |
| Constraint calc wrapped in a function | **Confirmed** | = `calc_constraints()`, §3. |
| Regridding: Sphere-in-Sphere / Box-in-Box | **Confirmed** | Box-in-Box is the same code path selected via `BSSN_BOX_TYPE` (0 = spherical/Euclidean, else box/L∞). See §5. |
| Regridding: WAMR on value/error of a constraint var | **Confirmed** | `isRemeshConstraint` (value) and `isReMeshWAMRConstraint` (error). See §5. |
| Regridding: relative-error param `Δf/(PAR+f)` instead of `Δf/(1+f)` | **NOT actually implemented** | A `BSSN_REL_ERR_MIN` param was added and read in, but its only call site is **commented out** (`dataUtils.cpp:364`). The effective wavelet normalization in `dendrolib` is `Δf / max(1.0, |f|)` with a hardcoded floor of `1.0`, and it is **identical** in the branch and master submodule pins. So this feature is *intended but unwired*. See §8. |

### Corrections to your recollection
- **`grad2_chi` square-root.** Commit `2f440d4` removed the sqrt ("not a 2-norm anymore"),
  but a later commit restored it. The **final branch state has the sqrt** (Frobenius norm of
  the Hessian of chi). So `grad2_chi` *is* a 2-norm again.
- **Relative-error feature** is not realized in code (above). If you want `Δf/(PAR+f)`, it has
  to be (re)written into `dendrolib`'s `WaveletEl::compute_wavelets_3D`.

---

## 2. Main routine — walltime, checkpoints, timing  (`bssngr_main.cpp`)

Self-contained additions to the evolution loop; easy to graft onto master's `main`.

- **Wall-time termination.** `start_time = MPI_Wtime()` captured pre-init; each step rank-0
  checks `(MPI_Wtime() - start_time)/60.0 > WALL_TIME` (minutes) and broadcasts a terminate
  flag. (`bssngr_main.cpp:430`, param `WALL_TIME`)
- **Guaranteed final checkpoint.** `already_checkpointed_in_this_it` flag; writes a final
  checkpoint on exit if one wasn't already written this interval.
- **Speed reporting.** Per-interval and end-of-run M/hour using `wall_interval_start`,
  `sim_interval_start`, `init_done_time`, `init_time`. (`bssngr_main.cpp:444-466`)
- **`printtime()` helper** + `start.at_now` sentinel file written at evolution start.
- **`bssn::TEMP_BSSN_STEP_VAL = ets->curr_step()`** each step — a global used by the VTU
  error-dump diagnostic in `dataUtils.cpp`. Marked TEMP/TODO-remove.

## 3. Constraint calculation wrapped for regridding  (`bssnCtx.cpp`, `bssnCtx.h`)

- **`BSSNCtx::calc_constraints()`** (`bssnCtx.cpp:577-648`, declared `bssnCtx.h:155`):
  unzips evolution vars, calls `physical_constraints()` per block, zips + ghost-syncs the
  constraint vars. Pattern mirrors `write_vtu()`.
- Called from `main` **before** `is_remesh()` (`bssngr_main.cpp:370`) so constraint-based
  refinement sees fresh values.
- **`is_remesh()` extended** (`bssnCtx.cpp:1160-1220`): unzips constraint vars and dispatches
  the 4 new refinement modes to the `dataUtils` functions in §5.
- **FEATURE — wavelet tolerance is re-read from the par file on restart.** The branch comments
  out *both* sites that restored `bssn::BSSN_WAVELET_TOL` from the checkpoint
  (branch `bssnCtx.cpp:929-930` and `:982`; master equivalents at `bssnCtx.cpp:1061-1062` and
  `:1188`). Master's stock behavior locks the wavelet tolerance to whatever it was when the
  first checkpoint was written, so it can never change across restarts. With the restore
  disabled, the par-file value wins — letting you tighten/loosen the wavelet tol when resuming.
  This is intentional and should be ported (cleanly, via a parameter — see RESYNC.md Step 4b),
  **not** discarded.

## 4. New constraint variables  (small edits — port verbatim)

Three new `VAR_CONSTRAINT` entries (`grDef.h:76-78`, names at `:88`), pointers in
`physcon.cpp:33-35`, equations in `physconeqs.cpp:701-705`:

- `C_GRAD_CHI` → `grad_chi = sqrt(Σ (∂_i χ)²)` — magnitude of ∇χ.
- `C_GRAD2_CHI` → `grad2_chi = sqrt(Σ (∂_ij χ)²)` (off-diagonals ×2) — Frobenius norm of the
  Hessian of χ. **(sqrt present in final state.)**
- `C_GRAD_GRAD2_CHI_EXPRESSION` → `grad2_chi / (χ·(1-χ)³)` (two alternative scalings left
  commented at `physconeqs.cpp:703-704`).

Also: `BSSN_CONSTRAINT_NUM_VARS` 6→9 (`parameters.h:37`); output-index list updated
(`parameters.cpp:138`). `bssn_constraints.h:47` swaps `exit(0)`→`MPI_Abort` on negative
metric determinant (robustness, not a feature).

## 5. New regridding / refinement methods  (`dataUtils.cpp` +~1000 lines, `dataUtils.h`)

All new, self-contained free functions; none exist in master. Header decls `dataUtils.h:10-18`.
`#define OCT_IGNORE 10u` and includes added at `dataUtils.cpp:31-40`.

**Geometry helpers** (`dataUtils.cpp:60-103`): `point_linf`,
`min_distance_cell_to_point_1D`, `min_distance_cell_to_point` (Euclidean vs L∞ via
`BSSN_BOX_TYPE` → this is the **Sphere-in-Sphere vs Box-in-Box** switch).

| Mode (enum) | Entry fn | Helper(s) | What it does |
|---|---|---|---|
| `SPHERE_IN_SPHERE` | `isRemeshSinS` (`:972`) | `isRemeshSinSHelper` (`:598`), `isRemeshSinSInitHelper` (`:519`) | Per-puncture concentric spheres/boxes; map cell→nearest puncture→target level from `BSSN_BOX_RADII_{1,2}`; split/coarse/no-change. |
| `SIS_OUT_WAMR_IN` | `isRemeshSiSCombination` (`:496`) | `isReMeshWAMRHelper` (`:690`) | SiS flags first; cells SiS marks `OCT_IGNORE` are filled by WAMR (SiS limits where WAMR may act). |
| `CONSTRAINT` | `isRemeshConstraint` (`:368`) | `isRemeshConstraintHelper` (`:394`) | Refine on **value** of `log10|grad2_chi/χ²|`; binary-search `BSSN_CHI_VALUES[]`→level; majority vote over DOFs. |
| `CONSTRAINT_ERROR` | `isReMeshWAMRConstraint` (`:105`) | `isReMeshWAMRConstraintHelper` (`:137`) | WAMR on **error** of `grad_grad2_chi_expression`; time-gated SiS→WAMR transition; SiS clamps WAMR's allowed level change (≤1 level); inner SiS region; **dumps wavelet error to VTU** for visualization. Latest commit forbids OCT_SPLIT from this path. |

`isReMeshWAMRHelper` also enforces per-BH near-field levels (`BSSN_BH{1,2}_AMR_R`,
`BSSN_BH{1,2}_MAX_LEV`) and a merged-BH separation tolerance (`BH_MERGED_SEP_TOL=0.1`).

## 6. New parameters  (`parameters.{h,cpp}`, `grUtils.cpp`)

Each parameter is read in **both** TOML (`parameters.cpp`) and JSON (`grUtils.cpp`) readers.

| Param | Type | Default | Purpose |
|---|---|---|---|
| `WALL_TIME` | double (read as int) | 1e300 | Wall-clock limit, **minutes**. |
| `BSSN_BOX_NUM_LEVELS[2]` | uint[2] | — | #sphere levels per BH. **Mandatory** for SPHERE_IN_SPHERE (else `MPI_Abort`). |
| `BSSN_BOX_RADII_1[]`, `BSSN_BOX_RADII_2[]` | double[≤20] | — | Sphere/box radii per BH. **Mandatory** for SPHERE_IN_SPHERE. |
| `BSSN_BOX_TYPE` | uint | 0 | 0 = sphere (Euclidean), else box (L∞). |
| `BSSN_MINDEPTH_SIS` | uint | 7 | Min level for SiS/BiB. |
| `BSSN_CHI_VALUES[]` | double[≤20] | — | grad2χ contour thresholds (constraint refinement). |
| `BSSN_CHI_NUM_VALUES` | uint | 9 | Size of above. |
| `BSSN_SIS_TO_CONSTRAINT_WAMR_TRANSITION_TIME` | double | 60.0 | Time to switch SiS→constraint-WAMR. |
| `BSSN_INNER_SIS_REGION_OUTER_BOUND` | double | 5.0 | Inner SiS region radius for CONSTRAINT_ERROR. |
| `BSSN_REL_ERR_MIN` | double | 1.0 | Intended `PAR` in `Δf/(PAR+f)` — **read but not wired** (§8). |
| `BSSN_REFINEMENT_NUM_MODES` | uint | 2 | Internal; #modes combined. No file parse. |
| `BSSN_REFINEMENT_MODE_COMBINATION_ORDER[2]` | uint[2] | — | Internal; outer→inner mode order. No file parse. |
| `TEMP_BSSN_STEP_VAL` | uint | 0 | Internal debug step counter (TODO remove). |

Const `BSSN_BOX_MAX_RADII = 20` bounds the array params (`parameters.h:346`).

## 7. `dendrolib` submodule changes

Branch pins `8654c4e` (dirty), master pins `07b5d37`. Commits the branch adds over master's pin:
`8654c4e new oct_ignore`, `1d71824 feat: refinement flags by eleid`, plus an autoformat
commit (most of the line-count is cosmetic). Substantive, feature-relevant pieces:

- **`OCT_IGNORE = 10u`** support in `mesh.h`/`mesh.tcc` — required by the `SIS_OUT_WAMR_IN`
  combination (SiS marks cells `OCT_IGNORE`, WAMR fills them).
  *Note:* the working tree has a **dirty edit removing `#define OCT_IGNORE` from `mesh.h`*
  (`dendrolib/include/mesh.h`), because the parent `dataUtils.cpp` now `#define`s it itself.
  However you slice it, `OCT_IGNORE=10u` must be visible to the combination logic.
- **Get-refinement-flags-by-eleid** — supports the refine-flag / error visualization.

When master's redesign pins a newer Dendro, these two capabilities must exist there (or be
re-added). The relative-error normalization in `WaveletEl::compute_wavelets_3D`
(`src/waveletRefEl.cpp`) is **unchanged** between the two pins — see §8.

## 8. Relative-error parameter — status: incomplete

- Effective formula in both pins: `wc = |Δf| / max(in_min, |f|)` with `in_min = 1.0`
  hardcoded (`dendrolib/src/waveletRefEl.cpp`). This is *not* `Δf/(1+f)` and not parameterized.
- The parent repo added `BSSN_REL_ERR_MIN` and reads it, but the only place it would be used
  (`dataUtils.cpp:364`, a `compute_wavelets_3D(... , BSSN_REL_ERR_MIN)` call) is **commented
  out**, and the submodule signature doesn't even take that argument.
- **To actually get `Δf/(PAR+f)`**: thread `BSSN_REL_ERR_MIN` into
  `WaveletEl::compute_wavelets_3D` and change `max(in_min,|f|)` → `(in_min + |f|)`
  (or `max(BSSN_REL_ERR_MIN,|f|)`), then pass it from the `dataUtils` refinement functions.

## 9. Noise to ignore when porting

- Removal of the **AEH apparent-horizon solver** and the startup banner in `bssngr_main.cpp`
  (large deletions — branch-specific cleanup, not features).
- Destructor debug `cout`s (`bssnCtx.cpp:59-67`); `getMPIRankGlobal()`→`getMPIRank()`;
  `start.at_now` sentinel.
  (NOTE: the commented-out `BSSN_WAVELET_TOL` checkpoint restore is **not** noise — it is a
  real feature; see §3.)
- New `.par.{toml,json}` test inputs (`pars/q2rf*`, `DeltaDeltaChiSegfault…`, etc.) and a
  stray `.patch` file — test fixtures, port only the ones you need.
