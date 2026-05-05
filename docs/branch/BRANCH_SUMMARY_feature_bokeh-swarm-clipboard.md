# Branch Summary: feature/bokeh-swarm-clipboard

## Metadata
- **Branch**: feature/bokeh-swarm-clipboard
- **Status**: open
- **Opened on**: 2026-05-05
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
P26-05-05
P26-05-05

## Accomplishments
- Opened branch **`feature/bokeh-swarm-clipboard`** to add Bokeh-backed Swarm-style **Clipboard** (and later **Helicorder**) beside existing matplotlib `swarmmpl` code.
- Audited **`ClipboardClass`** / **`Clipboard`** and **`SwarmClipboard`** in `vdapseisutils.core.swarmmpl.clipboard` plus dependencies (`TimeAxes`, **`prepare_waveform_series`**, **`compute_spectrogram`**).
- Drafted **`docs/plans/bokeh-swarm-clipboard-plan.md`** with phased milestones, Bokeh layout/glyph notes, risks, and open API questions aligned with the **`core.maps.bokeh`** precedent.

## Planned work
- Confirm parity target (**`SwarmClipboard`** first vs legacy **`ClipboardClass`**) and downsampling/export expectations (see plan **Open questions**).
- Implement Phase 0 scaffold (module path, optional `[bokeh]` extra alignment, minimal API sketch).
- Deliver Phase 1 waveform-only multi-panel with linked ranges and datetime/relative ticks; then spectrograms and overlays per plan checklist.

## Executed work
- Created git branch **`feature/bokeh-swarm-clipboard`** from **`main`** and regenerated **`docs/branch/BRANCH_TIMELINE.md`** via **`scripts/branch_docs.py post-checkout`**.
- Added **`docs/plans/bokeh-swarm-clipboard-plan.md`** and populated this branch summary for hook validation.

## Back-and-forth / iteration notes
- User asked to start with **Clipboard**; **Helicorder** remains explicit follow-on under the same branch initiative.

## Problems + resolutions
- Root **`__init__.py`** documents **`from vdapseisutils.swarmbk import Clipboard_bk`**, but **`swarmbk`** is not present in-tree—called out as repo hygiene / non-goal for Clipboard v1 in the plan (no code change yet).

## Validation
- Branch summary passes **`scripts/branch_docs.py`** metadata and accomplishments count; timeline regenerated.

## Final changelog-style outcome
- Pending implementation: Bokeh Clipboard module, tests, and gallery notebook per **`docs/plans/bokeh-swarm-clipboard-plan.md`**.
