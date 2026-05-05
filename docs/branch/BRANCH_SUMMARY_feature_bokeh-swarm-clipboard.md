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
- Drafted **`docs/plans/bokeh-swarm-clipboard-plan.md`** with phased milestones, Bokeh layout/glyph notes, risks, and API decisions (**`SwarmClipboard`** parity, **`vdapseisutils.core.swarmmpl.bokeh`**, HTML export in first drop, downsampling allowed with follow-up design discussion).

## Planned work
- Implement Phase 0 scaffold under **`vdapseisutils.core.swarmmpl.bokeh`** (class names, optional `[bokeh]` extra alignment, HTML **`save`** / standalone export on the first usable milestone).
- Deliver Phase 1 waveform-only multi-panel with linked ranges, datetime/relative ticks, and documented downsampling hooks; then spectrograms and overlays per plan checklist.
- Nail down downsampling defaults and user controls (algorithm, max points, per-trace vs global) after initial notebook iteration.

## Executed work
- Created git branch **`feature/bokeh-swarm-clipboard`** from **`main`** and regenerated **`docs/branch/BRANCH_TIMELINE.md`** via **`scripts/branch_docs.py post-checkout`**.
- Added **`docs/plans/bokeh-swarm-clipboard-plan.md`** and populated this branch summary for hook validation.
- Captured stakeholder answers on parity (**`SwarmClipboard`**), downsampling (allowed), notebook + HTML export, and package layout (**`core.swarmmpl.bokeh`**).

## Back-and-forth / iteration notes
- User asked to start with **Clipboard**; **Helicorder** remains explicit follow-on under the same branch initiative.
- Stakeholder locked **`SwarmClipboard`** parity, **`vdapseisutils.core.swarmmpl.bokeh`**, HTML **`save` on first usable drop**, and acceptable automatic downsampling with further design discussion.

## Problems + resolutions
- Root **`__init__.py`** documents **`from vdapseisutils.swarmbk import Clipboard_bk`**, but **`swarmbk`** is not present in-tree—called out as repo hygiene / non-goal for Clipboard v1 in the plan (no code change yet).

## Validation
- Branch summary passes **`scripts/branch_docs.py`** metadata and accomplishments count; timeline regenerated.

## Final changelog-style outcome
- Pending implementation: Bokeh Clipboard module, tests, and gallery notebook per **`docs/plans/bokeh-swarm-clipboard-plan.md`**.
