# CLAUDE.md — Dendro-GR

Numerical-relativity (BSSN) code built on the Dendro-5.01 adaptive-mesh library. The BSSN
solver lives in `BSSN_GR/` (`src/`, `include/`, `pars/`).

## dendrolib is a CMake FetchContent dependency (not a submodule on master)

Top-level `CMakeLists.txt` pulls dendrolib via `FetchContent`. Control which copy is used with:
- `DENDRO_dendrolib_DIR` — path to a **local** dendrolib checkout (`add_subdirectory`).
- `DENDRO_dendrolib_GIT_TAG` — branch/tag/commit (default `master`).
- `GIT_REPOSITORY` in `CMakeLists.txt` — default `https://github.com/paralab/Dendro-5.01`.

To build against a fork of dendrolib (e.g. one implementing `BSSN_REL_ERR_MIN`), point
`DENDRO_dendrolib_DIR` at a local clone or change the repo/tag.

> Note: the legacy `chi-tests-SiS-limit` branch instead vendored dendrolib as a git
> **submodule** with a custom pinned commit. That model is obsolete — do not reintroduce it.

## Parameters are table-driven (TOML)

`BSSN_GR/src/parameters.cpp` reads TOML via `requiredParsList` / `optionalParsList`
(vectors of `ParameterInformation`). **Add a new parameter = one table row** + a global decl
in `parameters.h` + definition in `parameters.cpp`. Fixed C-array params that don't fit the
generic `set_param` use a manual-indexing block (~`parameters.cpp:584`).
A separate **JSON** reader exists in `grUtils.cpp` (`readParamJSONFile`, manual `parFile.find`)
— update it only if JSON par files are needed; prefer TOML for new work.

## Key BSSN entry points

- `BSSNCtx::is_remesh()` (`bssnCtx.cpp`) — dispatches on `bssn::BSSN_REFINEMENT_MODE`
  (`enum RefinementMode` in `grDef.h`: `WAMR, EH, EH_WAMR, BH_LOC, BH_WAMR`).
- `BSSNCtx::compute_constraint_variables()` + `get_constraint_vars()` — compute/retrieve the
  constraint fields (used for constraint-based refinement and VTU output).
- Constraint variables: `enum VAR_CONSTRAINT` in `grDef.h`; computed in
  `physcon.cpp` (pointer setup) and `physconeqs.cpp` (equations); count is
  `BSSN_CONSTRAINT_NUM_VARS` in `parameters.h`.
- Main evolution loop: `bssngr_main.cpp`, `while (ets->curr_time() < bssn::BSSN_RK_TIME_END)`.

## Active effort: RIT feature resync

`chi-tests-SiS-limit` (legacy, never merged) holds RIT refinement/utility features. We are
**reimplementing** them on top of current `master` in a new branch **`rit_devel_next`**, not
merging git history.

- **`RESYNC.md`** — the step-by-step port plan (read it before touching resync work).
- **`CHI_TESTS_SIS_LIMIT_FEATURES.md`** — full inventory of what that branch added, with
  file:line references and verification notes.

Scope: only Sphere/Box-in-Box and the two constraint refinement modes are ported; the
SiS+WAMR *combination* mode (`SIS_OUT_WAMR_IN`) and its helpers/params are dropped. This
**renumbers `BSSN_REFINEMENT_MODE`** vs the legacy branch (Sphere-in-Sphere `4 → 5`, since
master inserted `BH_WAMR = 4`), so old par files need updating. The renumbering, dropped
features, and new parameters get a user-facing doc at `BSSN_GR/doc/REFINEMENT_AND_PARAMS_rit.md`
(see RESYNC.md Step 5b).

Feature summary being ported: Sphere-in-Sphere / Box-in-Box refinement, constraint-value and
constraint-error (chi-gradient) WAMR refinement (isolated in a **new** `refinement_sis.{cpp,h}`),
wall-time termination + checkpoint-on-terminate (into `bssngr_main.cpp`), three new chi-based
constraint variables, and the `BSSN_REL_ERR_MIN` relative-error parameter (formula lives in a
dendrolib fork).

## Conventions for resync edits

- Keep edits to existing files append-only / single-block; **do not reformat** — a large
  upstream batch is expected and must merge cleanly.
- Put substantial new logic in **new files**.
- Append new enum values (never renumber; `BH_WAMR = 4` is taken).
- Do **not** re-apply `chi-tests-SiS-limit` cleanup noise (AEH-solver removal, debug `cout`s,
  commented-out checkpoint-restore) — those are not features.
