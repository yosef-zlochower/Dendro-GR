# RESYNC plan — port `chi-tests-SiS-limit` features onto current `master`

**Target branch:** `rit_devel_next`, based on the **current** `master` (`b3261e2`).
**Source of features:** `chi-tests-SiS-limit` (merge-base `c63392e`). See
`CHI_TESTS_SIS_LIMIT_FEATURES.md` for the full feature inventory and line references.

**Guiding principle (req 9):** keep every edit to an *existing* master file as small and
append-only as possible, so the next large batch of upstream changes merges cleanly. Put new
logic in **new files** wherever an in-place edit would be a large/complicated diff (req 6).
Do **not** reformat existing files, and do **not** re-apply branch-only cleanup (AEH removal,
debug `cout`s, commented-out checkpoint restores) — those are noise, not features.

Master is in much better shape than the old merge-base for this work:
- Parameters are **table-driven** (`requiredParsList` / `optionalParsList` of
  `ParameterInformation`) — adding a param is one table row, not a manual reader block.
- `BSSNCtx::compute_constraint_variables()` and `get_constraint_vars()` **already exist** —
  these *are* the `calc_constraints()` wrapper the branch hand-rolled.
- dendrolib is consumed via CMake `FetchContent`, not a submodule (req 7).

---

## Step 0 — Create the branch

```
git checkout master && git pull
git checkout -b rit_devel_next
```

---

## Step 1 — New refinement functions in their own file (req 1, 6)

Create **`BSSN_GR/src/refinement_sis.cpp`** + **`BSSN_GR/include/refinement_sis.h`**
(namespace `bssn`). Move the branch's new free functions here verbatim, then fix to compile:

**Scope (decided):** keep only Sphere/Box-in-Box and the two constraint modes. The
SiS-outer/WAMR-inner *combination* mode and everything that exists only to serve it are
**abandoned**. Verified call graph (branch `dataUtils.cpp`):

**KEEP** → port to `refinement_sis.{cpp,h}`:

| Function(s) | Def line (branch) | Mode | Notes |
|---|---|---|---|
| `point_linf`, `min_distance_cell_to_point_1D`, `min_distance_cell_to_point` | 60–103 | (shared geometry) | used by SiS |
| `isRemeshSinS` | 1080 | SPHERE_IN_SPHERE | **self-contained** — calls no helper, computes flags inline |
| `isRemeshSinSInitHelper` | 627 | (shared) | used by *both* constraint functions for SiS level-limiting |
| `isReMeshWAMRConstraint` + `isReMeshWAMRConstraintHelper` | 213 / 245 | CONSTRAINT_ERROR | own inline `compute_wavelets_3D`; uses `OCT_IGNORE` internally (lines 231, 382, 384) |
| `isRemeshConstraint` + `isRemeshConstraintHelper` | 476 / 502 | CONSTRAINT (value) | ⚠️ suspected bug — see note below; keep code as-is for now |

**ABANDON** → do **not** port (all reachable only from the combination mode):

| Function | Def line | Reason |
|---|---|---|
| `isRemeshSiSCombination` | 604 | mode SIS_OUT_WAMR_IN — out of scope |
| `isRemeshSinSHelper` | 706 | sole caller is `isRemeshSiSCombination` (line 612) |
| `isReMeshWAMRHelper` | 798 | sole caller is `isRemeshSiSCombination` (line 613) |

Consequence: the combination-only parameters `BSSN_REFINEMENT_NUM_MODES` and
`BSSN_REFINEMENT_MODE_COMBINATION_ORDER` are referenced **only** inside the abandoned
`isRemeshSinSHelper`, so they are **dropped** (see Step 5).

> ⚠️ **`isRemeshConstraint` suspected bug.** A student suspects a bug in the value-based
> constraint method. Decision: **keep the code in `rit_devel_next` unchanged** (do not fix it
> as part of the resync) but add a `// TODO(rit): suspected bug — verify` marker and flag it in
> the validation checklist so it gets reviewed separately.

Fix-ups required:
- **`OCT_IGNORE`.** Still needed — but only by the *kept* `isReMeshWAMRConstraint`/Helper,
  which uses `OCT_IGNORE = 10u` as an internal "not in WAMR region" marker (not the combination
  meaning). `#define OCT_IGNORE 10u` at the top of `refinement_sis.cpp`, guarded by
  `#ifndef OCT_IGNORE` so it works regardless of the dendrolib version.
- **No WAMR-helper port.** Because `isReMeshWAMRHelper` is abandoned, there is no
  reuse-vs-copy decision to make. The kept constraint-WAMR path does its own wavelet
  computation inline and does **not** depend on master's `isReMeshWAMR`/`addRemeshWAMR`.
- **REL_MIN_ERROR call site (req 8).** Keep `bssn::BSSN_REL_ERR_MIN` and pass it into the
  wavelet computation (the branch left this commented at `dataUtils.cpp:364`). The call must
  match the **forked** dendrolib's `WaveletEl::compute_wavelets_3D(..., rel_min)` signature, and
  is **unconditional** — there is no feature-macro fallback. If the supplied dendrolib lacks the
  extra arg, the build **fails by design**, so `rel_min` can never be silently dropped to the
  stock 4-arg form (the user can't be tricked into thinking the floor is active when it isn't).
- **VTU error dump.** `isReMeshWAMRConstraintHelper` writes the wavelet error to VTU for
  visualization (branch lines 347–359) using `TEMP_BSSN_STEP_VAL`, `BSSN_VTU_FILE_PREFIX`,
  etc. Keep it but gate it behind a parameter (e.g. only when an output flag is set) so it
  isn't a hidden per-remesh I/O cost.

Register the new `.cpp` in **`BSSN_GR/CMakeLists.txt`** (add to the existing source list — one
line). Add `#include "refinement_sis.h"` only where needed (Step 4).

`refinement_sis.h` declares the public entry points consumed by `is_remesh()`:
`isRemeshSinS`, `isRemeshConstraint`, `isReMeshWAMRConstraint` (no `isRemeshSiSCombination`).

---

## Step 2 — Utility functionality directly into `main` (req 2)

Edit **`BSSN_GR/src/bssngr_main.cpp`** inside the existing
`while (ets->curr_time() < bssn::BSSN_RK_TIME_END)` loop (master line ~410). **Add, do not
restructure**, and **leave the AEH solver block in place**.

1. Before the loop: `const double start_time = MPI_Wtime();` and
   `bool already_checkpointed_in_this_it = false;` plus speed-tracking anchors
   (`wall_interval_start`, `sim_interval_start`, `init_done_time`, `init_time`).
2. Inside the loop, rank-0 wall-time check:
   `if ((MPI_Wtime() - start_time) / 60.0 > bssn::WALL_TIME) { <broadcast terminate> }`
   (WALL_TIME is in **minutes**). Broadcast the flag to all ranks and `break`.
3. **Checkpoint on terminate** + guaranteed **final checkpoint**: set
   `already_checkpointed_in_this_it` where the normal checkpoint is written (master ~line
   602); after the loop, write one final checkpoint if the flag is unset.
4. Optional: per-interval and end-of-run **M/hour speed report**, `printtime()` helper, and
   the `start.at_now` sentinel — these are convenience-only; include if wanted, they touch
   only `main`.

Keep all of this in one contiguous, clearly-commented region so the diff is one block.

---

## Step 3 — New constraint variables (req 3, small in-place edits)

These are intentionally tiny, append-only edits to existing files:

- **`BSSN_GR/include/grDef.h`** — append to `enum VAR_CONSTRAINT` (after `C_PSI4_IMG`):
  `C_GRAD_CHI`, `C_GRAD2_CHI`, `C_GRAD_GRAD2_CHI_EXPRESSION`; append their names to
  `BSSN_CONSTRAINT_VAR_NAMES[]`.
- **`BSSN_GR/include/parameters.h`** — bump `BSSN_CONSTRAINT_NUM_VARS` 6 → 9.
- **`BSSN_GR/src/parameters.cpp`** — extend `BSSN_VTU_OUTPUT_CONST_INDICES` default to length 9.
- **`BSSN_GR/src/physcon.cpp`** — add 3 output pointers next to the existing
  `psi4_real/psi4_img` (branch `physcon.cpp:33–35`).
- **`BSSN_GR/src/physconeqs.cpp`** — add 3 equations next to the psi4 block
  (master ~line 1129); from the branch (`physconeqs.cpp:701–705`):
  - `grad_chi  = sqrt(Σ (∂_i χ)²)`
  - `grad2_chi = sqrt(Σ (∂_ij χ)²)` (off-diagonals ×2)  ← **keep the sqrt** (final branch state)
  - `grad_grad2_chi_expression = grad2_chi / (χ·(1-χ)³)`
  Ensure the required `grad_*_chi` / `grad2_*_chi` derivative locals are computed (they exist
  in the constraint-derivatives path used for the other constraints).

If the derivative bookkeeping makes `physconeqs.cpp` messy, factor the 3 lines into a small
helper in a new `constraint_chi.inc`/`.h` included at that point (req 6) — but the straight
3-line insert is expected to be clean.

---

## Step 4 — Wire constraint-based regrid into `is_remesh()` (req 4)

Edit **`BSSN_GR/src/bssnCtx.cpp`** `BSSNCtx::is_remesh()` (master ~line 1444). Append new
`else if` branches after the existing `BH_WAMR` branch, dispatching to the Step-1 functions:

```
} else if (bssn::BSSN_REFINEMENT_MODE == bssn::RefinementMode::SPHERE_IN_SPHERE) { ... }
} else if (... CONSTRAINT) { ... }        // value-based isRemeshConstraint
} else if (... CONSTRAINT_ERROR) { ... }  // constraint-based WAMR isReMeshWAMRConstraint
```
(No `SIS_OUT_WAMR_IN` branch — that mode is abandoned.)

For the modes that read constraint variables (`CONSTRAINT`, `CONSTRAINT_ERROR`):
- **Reuse master's existing machinery** instead of the branch's hand-rolled unzip: call
  `this->compute_constraint_variables();` then `this->get_constraint_vars();` and unzip that
  DVec to get the `unzippedcVec` the new functions expect. Gate the compute call so it only
  runs for the constraint modes (avoid the cost for plain WAMR/SiS).
- Pass the constraint-variable indices (`C_GRAD2_CHI`, `C_CHI`/chi source,
  `C_GRAD_GRAD2_CHI_EXPRESSION`) into the new functions.

`#include "refinement_sis.h"` at the top of `bssnCtx.cpp` (one line).

Do **not** add the branch's separate `calc_constraints()` method — master already provides the
equivalent.

---

## Step 4b — Let the wavelet tolerance change across restarts (req 2/4)

**Feature:** master's checkpoint restore overwrites `bssn::BSSN_WAVELET_TOL` with the
checkpointed value at **two** sites — `bssnCtx.cpp:1061-1062` (multi-rank restore) and
`:1188` (single-rank restore) — so the wavelet tolerance is permanently locked to its value at
the first checkpoint and can never be tuned on resume. The branch disabled both (by commenting
them out) so the par-file value wins.

**Reimplement cleanly — do not just comment the lines out.** Add an optional boolean parameter
and guard both restore sites with it, so the intent is explicit and self-documenting:

```cpp
if (bssn::BSSN_RESTORE_WAVELET_TOL_FROM_CHECKPOINT)
    bssn::BSSN_WAVELET_TOL = checkPoint["DENDRO_TS_WAVELET_TOLERANCE"];
// else: keep the value read from the par file (allows changing it on restart)
```

- Register `BSSN_RESTORE_WAVELET_TOL_FROM_CHECKPOINT` (bool) in `optionalParsList` (Step 5).
- **Default `false`** (decided — par-file wins, the desired feature). Setting it `true`
  reproduces exact stock-master restart behavior, so strict-compat users can opt back in.
- ⚠️ This is the one place the backward-compat invariant (Step 9) is intentionally relaxed: a
  master par file *resumed from a checkpoint* will now take its wavelet tol from the par file
  rather than the checkpoint. In practice identical unless the user changed the tol between
  runs. Document it in Step 5b. (Keep writing `DENDRO_TS_WAVELET_TOLERANCE` into the checkpoint
  at `bssnCtx.cpp:942` either way, so checkpoints stay forward/backward readable.)

---

## Step 5 — New parameters (req 5)

Declare globals in **`parameters.h`**, define in **`parameters.cpp`**, then register:

- **Scalar params** → one row each in `optionalParsList` (master `parameters.cpp:470`):
  `WALL_TIME` (1e300), `BSSN_MINDEPTH_SIS` (7), `BSSN_BOX_TYPE` (0), `BSSN_CHI_NUM_VALUES` (9),
  `BSSN_SIS_TO_CONSTRAINT_WAMR_TRANSITION_TIME` (60.0),
  `BSSN_INNER_SIS_REGION_OUTER_BOUND` (5.0), `BSSN_REL_ERR_MIN` (1.0),
  `BSSN_RESTORE_WAVELET_TOL_FROM_CHECKPOINT` (bool, **false** — see Step 4b).
  - **Dropped:** `BSSN_REFINEMENT_NUM_MODES` and `BSSN_REFINEMENT_MODE_COMBINATION_ORDER` are
    used only by the abandoned combination mode — do **not** port them.
  - `TEMP_BSSN_STEP_VAL` is debug-only (for the VTU error dump); include as a plain global
    only if the VTU dump is kept, otherwise drop it.
- **Array / vector params** (`BSSN_BOX_NUM_LEVELS[2]`, `BSSN_BOX_RADII_1[]`,
  `BSSN_BOX_RADII_2[]`, `BSSN_CHI_VALUES[]`): the table handles `std::vector<int/double>`
  cleanly, so prefer declaring these as `std::vector<...>` and adding table rows. If they must
  stay fixed C arrays, use the existing "trickier params" manual-indexing block
  (`parameters.cpp:584`) — model on how `BSSN_REFINE_VARIABLE_INDICES` is handled.
- **Mandatory-for-SiS validation.** The branch `MPI_Abort`ed if `BOX_NUM_LEVELS`/`BOX_RADII_*`
  were missing under `SPHERE_IN_SPHERE`. Keep these in `optionalParsList` (so non-SiS runs are
  unaffected) and instead validate-then-abort **inside `is_remesh()`/`init_grid()`** when the
  mode is selected. Keeps the param table uniform.
- Add the three enum values to **`grDef.h`** `enum RefinementMode`, **appended after
  `BH_WAMR`**: `SPHERE_IN_SPHERE, CONSTRAINT, CONSTRAINT_ERROR` (no `SIS_OUT_WAMR_IN`).
  ⚠️ Do **not** reuse value 4 — master already assigns `BH_WAMR = 4`. Resulting numbering is
  `SPHERE_IN_SPHERE = 5, CONSTRAINT = 6, CONSTRAINT_ERROR = 7`. **This renumbers
  `BSSN_REFINEMENT_MODE` relative to the legacy branch** — must be documented (Step 5b).
- **JSON reader (`grUtils.cpp`)** is still manual (`parFile.find`). Mirror the new params there
  only if JSON par files are needed; otherwise standardize new par files on TOML and note JSON
  is not updated. (Lower priority.)

---

## Step 5b — Document the parameter & refinement-mode changes (user-facing)

Produce a **user-facing doc** so existing par files can be migrated. Create
**`BSSN_GR/doc/REFINEMENT_AND_PARAMS_rit.md`** (new file) and cross-link it from
`BSSN_GR/ReadMe.md` (one line). It must cover:

**(a) `BSSN_REFINEMENT_MODE` renumbering — migration-critical.** Master inserted `BH_WAMR = 4`
and the SiS-combination mode was dropped, so the new-vs-legacy numbering differs. Include this
exact table and a prominent warning that old par files must be updated:

| Mode | Legacy `chi-tests-SiS-limit` value | New `rit_devel_next` value |
|---|---|---|
| `WAMR` / `EH` / `EH_WAMR` / `BH_LOC` | 0 / 1 / 2 / 3 | 0 / 1 / 2 / 3 (unchanged) |
| `BH_WAMR` | — (did not exist) | 4 |
| `SPHERE_IN_SPHERE` | 4 | **5** ← changed |
| `SIS_OUT_WAMR_IN` | 5 | **removed** |
| `CONSTRAINT` | 6 | 6 (unchanged) |
| `CONSTRAINT_ERROR` | 7 | 7 (unchanged) |

> ⚠️ A legacy par file with `BSSN_REFINEMENT_MODE = 4` (Sphere-in-Sphere) must change to `5`.

**(b) Dropped features** (state explicitly that these no longer exist):
- Refinement mode `SIS_OUT_WAMR_IN` (the SiS-outer / plain-WAMR-inner combination).
- Parameters `BSSN_REFINEMENT_NUM_MODES` and `BSSN_REFINEMENT_MODE_COMBINATION_ORDER`
  (only the dropped combination mode used them).

**(c) New parameters reference table** — for each: name, type, default, **units**, required vs
optional, and which mode/feature consumes it. Cover:
`WALL_TIME` (double, **minutes**), `BSSN_MINDEPTH_SIS`, `BSSN_BOX_TYPE` (0 = sphere/Euclidean,
else box/L∞), `BSSN_BOX_NUM_LEVELS[2]`, `BSSN_BOX_RADII_1[]`, `BSSN_BOX_RADII_2[]`
(last three **required** under `SPHERE_IN_SPHERE`), `BSSN_CHI_VALUES[]`,
`BSSN_CHI_NUM_VALUES`, `BSSN_SIS_TO_CONSTRAINT_WAMR_TRANSITION_TIME`,
`BSSN_INNER_SIS_REGION_OUTER_BOUND`, `BSSN_REL_ERR_MIN`.

**(d) Changed semantics / gotchas:**
- `WALL_TIME` is in **minutes** (legacy code at one point used seconds).
- `BSSN_REL_ERR_MIN` sets the relative-wavelet-error floor (`Δf / max(PAR, |f|)`, i.e. the
  wavelet normalization uses `max(BSSN_REL_ERR_MIN, |f|)` in the denominator — **not** the
  additive `PAR + |f|`); default `1.0` reproduces stock behavior. It has **no effect unless
  built against the forked dendrolib** that accepts the extra arg (Step 8). Document that
  dependency.
- `BSSN_CONSTRAINT_NUM_VARS` goes 6 → 9 (three new chi constraints). Any par file that sets
  `BSSN_VTU_OUTPUT_CONST_INDICES` or depends on constraint-output ordering must be reviewed.
- **Restart semantics changed:** the wavelet tolerance is now taken from the par file on resume
  (not the checkpoint). Document `BSSN_RESTORE_WAVELET_TOL_FROM_CHECKPOINT` (default `false`)
  and that setting it `true` restores stock behavior.
- Note the **suspected bug** in the value-based `CONSTRAINT` mode (`isRemeshConstraint`) so
  users treat its output with caution.

**(e) New refinement modes** — a short prose description of `SPHERE_IN_SPHERE`, `CONSTRAINT`,
and `CONSTRAINT_ERROR`, each with a minimal example TOML par snippet showing the required
parameters.

Keep the doc in-repo (travels with the branch). Also add a one-paragraph entry to the
top-level `CHANGELOG`/`ReadMe.md` summarizing the mode renumbering and dropped features.

---

## Step 6 — New-file vs in-place summary (req 6)

| Concern | Approach |
|---|---|
| SiS / constraint refinement functions | **New file** `refinement_sis.{cpp,h}` |
| Walltime / checkpoint-on-terminate / speed report | **In-place** in `bssngr_main.cpp` (one block) |
| New constraint variables | **In-place**, 3 tiny inserts (`grDef.h`, `physcon.cpp`, `physconeqs.cpp`) |
| `is_remesh()` dispatch | **In-place**, appended `else if` branches |
| New parameters | **In-place** table rows + global decls |
| Optional VTU error helper | keep in `refinement_sis.cpp`, parameter-gated |
| Parameter / mode-renumbering docs | **New file** `BSSN_GR/doc/REFINEMENT_AND_PARAMS_rit.md` (Step 5b) |

---

## Step 7 — dendrolib handling on master (req 7)

Master fetches dendrolib via CMake `FetchContent` (top-level `CMakeLists.txt:127–158`). Three
knobs already exist:
- `DENDRO_dendrolib_DIR` — path to a **local** dendrolib checkout (`add_subdirectory`).
- `DENDRO_dendrolib_GIT_TAG` — branch/tag/commit (default `master`).
- `GIT_REPOSITORY` — currently `https://github.com/paralab/Dendro-5.01`.

For the fork (Step 8), prefer pointing `DENDRO_dendrolib_DIR` at an **already-checked-out**
clone of your fork — this is the intended mechanism for using a specific local dendrolib:

```
cmake -DDENDRO_dendrolib_DIR=/path/to/Dendro-5.01-fork  <other flags>  ..
```

When that variable names a directory, the build does `add_subdirectory(${DENDRO_dendrolib_DIR}
dendrolib)` (`CMakeLists.txt:142-144`) and skips `FetchContent` entirely — **no clone, no
network, no `GIT_TAG`**. It compiles your working tree in place, so you can edit
`compute_wavelets_3D`, rebuild, and iterate. (Alternatively change `GIT_REPOSITORY` to the fork
URL and set `DENDRO_dendrolib_GIT_TAG` to the fork branch, but that re-clones into the build
tree and is worse for active development.)

There is **no submodule** to update — ignore the branch's `dendrolib` submodule pin entirely.
The branch's `OCT_IGNORE` and refine-flags-by-eleid changes either come from the fork or are
handled locally (Step 1’s `#ifndef OCT_IGNORE`).

---

## Step 8 — REL_MIN_ERROR / forked dendrolib (req 8)

Keep `BSSN_REL_ERR_MIN` end-to-end:
1. Param declared/registered (Step 5).
2. Threaded from the Step-1 refinement functions into `WaveletEl::compute_wavelets_3D`.
3. The relative-wavelet floor lives in **your dendrolib fork**, not in this repo. **Decision:**
   the fork keeps the existing `max`-based floor and only *parametrizes* it —
   `Δf / max(rel_min, |f|)` with `rel_min = BSSN_REL_ERR_MIN` — rather than switching to the
   additive `Δf / (PAR + |f|)` form the legacy branch intended (`max` judged more correct).
   `rel_min = 1.0` reproduces the stock `Δf / max(1.0, |f|)` exactly.
4. The call site (`refinement_sis.cpp`) is **unconditional** — it always passes
   `bssn::BSSN_REL_ERR_MIN` as the trailing arg. **There is no feature-macro fallback** (the old
   `#ifdef DENDRO_HAS_REL_ERR_MIN` / stock-4-arg `#else` was removed). Rationale: if the supplied
   dendrolib doesn't accept the extra arg, the build should **fail loudly here** rather than
   silently compiling the 4-arg form — otherwise a user could run believing the relative-error
   floor is active when it was quietly dropped. Building against the dendrolib fork (Step 7) is
   therefore a hard requirement of this branch.

### Fork recipe (the concrete plumbing)

**(a) Signature — add a defaulted parameter (DONE).** Both
`WaveletEl::compute_wavelets_3D` overloads now take a trailing `double rel_min = 1.0` (default
in the header decl). The hardcoded `const double in_min = 1.0;` was removed and all
`max(in_min, |f|)` sites renamed to `max(rel_min, |f|)`. Defaulting to `1.0` means:
- existing **4-arg / 6-arg** callers still compile and reproduce stock `Δf / max(1.0, |f|)`
  behavior — so the unguarded calls in `dataUtils.cpp:966` and `:1556` are unaffected;
- the floor stays `Δf / max(rel_min, |f|)` (kept the `max` form — **not** additive), so
  `rel_min = 1.0` is the identity case.

**(b) Thread-safe overload — patch both.** New Dendro-5.01 ships a thread-safe variant of the
same routine. Add the new parameter to **both** overloads with the same default, or refinement
results will silently diverge depending on which path the (possibly OpenMP-parallel) block loop
takes. Confirm which one the `wrefEl->compute_wavelets_3D(...)` member call in
`refinement_sis.cpp` actually resolves to before assuming only one needs the change.

**(c) No capability macro — the contract is the signature itself.** Dendro-GR calls the 5-arg
`compute_wavelets_3D` unconditionally (Step 8 item 4), so there is nothing to advertise and no
`DENDRO_HAS_REL_ERR_MIN` to define. Building with `-DDENDRO_dendrolib_DIR=/path/to/fork`
(Step 7) just works; building against a dendrolib whose `compute_wavelets_3D` lacks the trailing
`rel_min` arg fails to compile at the call site — which is the intended guardrail, not a
regression to work around.

---

## Step 9 — Keep diffs merge-friendly (req 9)

- Touch existing files only with append-only / single-block edits; never reformat.
- All substantial logic isolated in `refinement_sis.{cpp,h}`.
- Enum additions appended (no renumbering).
- Parameters added as table rows.
- Do **not** reintroduce branch-only removals (AEH solver, banners, debug prints).
  (The disabled `BSSN_WAVELET_TOL` checkpoint restore is the exception — it is a real feature,
  ported properly in Step 4b, not discarded.)

**Backward-compatibility invariant (must hold):** an *unmodified* current-master par file must
run on `rit_devel_next` unchanged. This is true **only if**:
- new `RefinementMode` values are **appended** after `BH_WAMR` (existing 0–4 unchanged);
- every new parameter is **optional** (`optionalParsList`) with a safe default — never added to
  `requiredParsList`, and no existing default is changed;
- new constraint vars are **appended** to `VAR_CONSTRAINT` (indices 0–5 unchanged); the VTU
  constraint output stays gated by `BSSN_NUM_CONST_VARS_VTU_OUTPUT` (default 1), so output for
  an existing par file is byte-for-byte identical;
- the SiS-"mandatory" `BSSN_BOX_*` abort fires only when `SPHERE_IN_SPHERE` is selected.
Breaking any of these breaks master par-file compatibility.
- **One intentional exception:** on *restart from a checkpoint*, the wavelet tolerance now comes
  from the par file, not the checkpoint (Step 4b). Identical unless the user changed the tol
  between runs; opt back into stock behavior with
  `BSSN_RESTORE_WAVELET_TOL_FROM_CHECKPOINT = true`.

---

## Step 10 — Validation checklist

- [x] `rit_devel_next` branches from current `master`.
- [ ] Builds against the **dendrolib fork** (whose `compute_wavelets_3D` takes `rel_min`):
      `bssnSolver` links cleanly. Note: the `rel_min` call is now **unconditional** (no
      `DENDRO_HAS_REL_ERR_MIN` guard), so building against stock dendrolib is expected to FAIL
      at the `compute_wavelets_3D` call — that is the intended guardrail.
- [ ] Existing refinement modes (WAMR/EH/EH_WAMR/BH_LOC/BH_WAMR) unchanged in behavior.
- [ ] **An unmodified current-master par file runs on `rit_devel_next` with identical results**
      (backward-compat invariant — see Step 9).
- [ ] A `SPHERE_IN_SPHERE` TOML par file runs; missing `BOX_*` params abort with a clear msg.
- [ ] A `CONSTRAINT` / `CONSTRAINT_ERROR` par file runs and refines on the new chi constraints.
- [ ] `isRemeshConstraint` (value-based `CONSTRAINT`) carries the `// TODO(rit): suspected bug`
      marker; bug review tracked separately (not fixed in this resync).
- [ ] Confirmed `isReMeshWAMRHelper`, `isRemeshSinSHelper`, and `isRemeshSiSCombination` are
      **not** ported, and combination params (`BSSN_REFINEMENT_NUM_MODES`,
      `BSSN_REFINEMENT_MODE_COMBINATION_ORDER`) are dropped.
- [ ] `WALL_TIME` (minutes) triggers terminate + final checkpoint; checkpoint-on-terminate works.
- [ ] Resume from a checkpoint with a **changed** `BSSN_WAVELET_TOL` in the par file actually
      uses the new value (Step 4b); `BSSN_RESTORE_WAVELET_TOL_FROM_CHECKPOINT = true` restores old.
- [ ] New constraint vars are selectable for VTU output (via `BSSN_VTU_OUTPUT_CONST_INDICES` +
      `BSSN_NUM_CONST_VARS_VTU_OUTPUT`) and look sane; default output for old par files unchanged.
- [ ] Port the few needed test par files from the branch (`pars/q2rf*`, etc.); convert to TOML,
      and **update `BSSN_REFINEMENT_MODE`** values to the new numbering (e.g. SiS 4 → 5).
- [ ] `BSSN_GR/doc/REFINEMENT_AND_PARAMS_rit.md` written (mode-renumber table, dropped features,
      new-parameter reference, changed semantics) and cross-linked from `BSSN_GR/ReadMe.md`.
