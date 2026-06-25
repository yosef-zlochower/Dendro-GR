# Constraint-based refinement — timing & caching findings (RIT)

Engineering notes for the constraint-based refinement modes (`CONSTRAINT`,
`CONSTRAINT_ERROR`) on branch `rit_devel_next`. Covers two related defects in how
derivative pointers and constraint data are sequenced around grid changes. Line
numbers are approximate — verify against current source.

Related docs: `REFINEMENT_AND_PARAMS_rit.md`, `RESYNC.md`, `CHI_TESTS_SIS_LIMIT_FEATURES.md`.

---

## 1. Null derivative-pointer segfault during init convergence (commit `c1fc75d`)

**Status:** fixed, reviewed correct.

**Root cause.** The global derivative function pointers `deriv_x` / `deriv_xx` / …
(declared in `derivs.h`, assigned by `set_appropriate_derivs()` in `derivs.cpp`) are
**null until set**. The initial grid-convergence `do`-loop in `BSSNCtx::initialize()`
calls `is_remesh()` each iteration; for `CONSTRAINT` / `CONSTRAINT_ERROR` that calls
`compute_constraint_variables()` → `physical_constraints()` →
`#include "constraint_derivs.h"` (`BSSN_GR/scripts/`), which invokes `deriv_xx` etc.
Pre-fix, `set_appropriate_derivs()` ran only *after* the loop (end of `initialize()`,
~line 545) and in `restore_checkpt()` (~line 1440) → null-pointer call → segfault.
`deriv_xx` is the 2nd-derivative call the commit message refers to (grad²χ).

**Fix (two parts in `bssnCtx.cpp`):**
1. **(Root fix)** Call `set_appropriate_derivs(BSSN_PADDING_WIDTH)` *before* the
   convergence loop in `initialize()` (~line 411). Idempotent and safe; now also still
   called at ~545 (harmless redundancy).
2. **(Optimization)** Split the combined `CONSTRAINT || CONSTRAINT_ERROR` branch in
   `is_remesh()`. For `CONSTRAINT_ERROR`, only call `compute_constraint_variables()`
   when `BSSN_CURRENT_RK_COORD_TIME > BSSN_SIS_TO_CONSTRAINT_WAMR_TRANSITION_TIME`,
   matching the internal guard in `isReMeshWAMRConstraint()`
   (`refinement_sis.cpp` ~line 83). `CONSTRAINT` (value mode) stays unconditional
   because `isRemeshConstraint` always reads constraint data.

**Caveat — transition time may be negative.** `BSSN_SIS_TO_CONSTRAINT_WAMR_TRANSITION_TIME`
is user-set; a negative or zero value ("use constraint-error WAMR from t=0") is valid.
At init, `BSSN_CURRENT_RK_COORD_TIME` is `0`, so `0 > transition` is **true** for a
negative transition → the `CONSTRAINT_ERROR` path *does* compute constraints during
init. Part 2's time-gate therefore does **not** shield `CONSTRAINT_ERROR` in that case;
only Part 1 prevents the segfault. **Part 1 is the essential fix.**

---

## 2. Constraint-variable cache goes stale across `grid_transfer`

**Status:** fix applied on `rit_devel_next` (separate from `c1fc75d`).

**Root cause.** `compute_constraint_variables()` caches via the flag
`m_bConstraintsComputed`: early-returns if true, sets true at the end. The flag is
reset only in `prepare_for_next_iter()` (`bssnCtx.h`), which runs once per **main**
evolution-loop iteration — **not** inside the init convergence loop.

`BSSNCtx::grid_transfer()` (`bssnCtx.cpp` ~1670) destroys and recreates the constraint
vectors (`CPU_CV`, `CPU_CV_UZ_IN`) **empty** for the new mesh; only the evolution vars
(`CPU_EV`) are actually grid-transferred. It did not invalidate the cache flag, so:

- **Init convergence loop:** after the first remesh, `is_remesh()` for constraint modes
  reads the wiped (recreated-empty) buffer because the cached flag skips recompute.
  Affects `CONSTRAINT` (value mode) **always**, and `CONSTRAINT_ERROR` whenever the
  transition time is `<= 0` (see §1 caveat).
- **Main loop:** on a remesh step the post-remesh constraint "refresh … for potential
  RHS updates" call (`bssngr_main.cpp` ~561) **silently no-ops**, because the flag is
  already true from the pre-remesh `is_remesh()` (or an earlier output-driven compute
  at ~443/466). It is never re-run before `prepare_for_next_iter()` resets the flag.
- **Not a crash:** the recreated buffer is correctly sized for the new mesh, so it is a
  *wrong-values* bug, not out-of-bounds.

**Fix.** Reset `m_bConstraintsComputed = false;` inside `grid_transfer()`, right after
the constraint vectors are recreated. `grid_transfer` is the CRTP chokepoint
(base `Ctx::grid_transfer` → `asLeaf().grid_transfer()` in
`Dendro-5.01/ODE/include/ctx.h`) reached by both the init loop (direct call at
`bssnCtx.cpp` ~461) and the main loop (`remesh_and_gridtransfer` →
`bssngr_main.cpp` ~504), so one line covers every grid change. Unconditionally safe:
there is no case where you want `computed = true` to survive a grid change.

**Intended side effect (not a regression).** Runs that compute constraints earlier in a
main-loop iteration (e.g. for VTU output) *and* remesh on the same step will now
actually run the line-561 refresh instead of skipping it — restoring the behavior the
code comment already asks for. Cost: one constraint recompute on remesh steps; zero
cost for runs that never request constraints.

---

## 3. Still open

- **Suspected bug in `isRemeshConstraint`** (value-based `CONSTRAINT` mode) — carries a
  `TODO(rit)` marker in `refinement_sis.cpp`; tracked separately, not addressed by the
  fixes above. Treat `CONSTRAINT`-mode output with caution.
