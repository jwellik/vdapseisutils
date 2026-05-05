# Branch Summary: feature/bokeh-swarm-clipboard

## Metadata
- **Branch**: feature/bokeh-swarm-clipboard
- **Status**: open
- **Opened on**: 2026-05-05
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
- **Last updated**: 2026-05-05

## Accomplishments
- Opened branch **`feature/bokeh-swarm-clipboard`** to add Bokeh-backed Swarm-style **Clipboard** (and later **Helicorder**) beside existing matplotlib `swarmmpl` code.
- Audited **`ClipboardClass`** / **`Clipboard`** and **`SwarmClipboard`** in `vdapseisutils.core.swarmmpl.clipboard` plus dependencies (`TimeAxes`, **`prepare_waveform_series`**, **`compute_spectrogram`**).
- Extended **`docs/plans/bokeh-swarm-clipboard-plan.md`** with **`gallery/SwarmMPL/Clipboard_tutorial_bokeh.ipynb`** (**Examples 1–3** mirroring **`Clipboard_Tutorial_A.ipynb`**) and **deferred downsampling** (“for meow”) until post‑MVP.
- Landed **`SwarmClipboardBk`** (**Phase 0–1**): HTML **`save()`**, **`examples/swarm_clipboard_minimal.py --save-bokeh-html`**, **`tick_type`** aliases (**absolute**/**datetime** vs **relative** seconds + **`sync_waves`** range linking), tests **`tests/test_swarm_clipboard_bokeh.py`**.

## Planned work
- Phase 2–4: spectrogram modes (**`wg`**/**`g`**), overlays (**`axvline`**, catalog), **`gallery/SwarmMPL/Clipboard_tutorial_bokeh.ipynb`**.

## Executed work
- Created git branch **`feature/bokeh-swarm-clipboard`** from **`main`** and regenerated **`docs/branch/BRANCH_TIMELINE.md`** via **`scripts/branch_docs.py post-checkout`**.
- Added **`docs/plans/bokeh-swarm-clipboard-plan.md`** and populated this branch summary for hook validation.
- Captured stakeholder answers on parity (**`SwarmClipboard`**), notebook + HTML export, package layout (**`core.swarmmpl.bokeh`**); downsampling policy explicitly deferred per latest guidance.
- Implemented Phase 0 package **`core/swarmmpl/bokeh/`**, smoke tests, example flag, and marked Phase 0 checklist complete in the plan.
- Completed Phase 1 waveform **`tick_type`** / **`sync_waves`** behavior and renamed tests to **`tests/test_swarm_clipboard_bokeh.py`**.

## Back-and-forth / iteration notes
- User asked to start with **Clipboard**; **Helicorder** remains explicit follow-on under the same branch initiative.
- Stakeholder locked **`SwarmClipboard`** parity, **`vdapseisutils.core.swarmmpl.bokeh`**, HTML **`save`** on first usable drop; requested **`Clipboard_tutorial_bokeh`** with Tutorial A **Examples 1–3**; **downsampling** discussion deferred for meow.

## Problems + resolutions
- Root **`__init__.py`** documents **`from vdapseisutils.swarmbk import Clipboard_bk`**, but **`swarmbk`** is not present in-tree—called out as repo hygiene / non-goal for Clipboard v1 in the plan (no code change yet).

## Validation
- Branch summary passes **`scripts/branch_docs.py`** metadata and accomplishments count; timeline regenerated.
- **`pytest -q tests/test_swarm_clipboard_bokeh.py`** passes.

## Final changelog-style outcome
- Phase 0–1 on branch: **`SwarmClipboardBk`** + HTML **`save`** + waveform tick/sync coverage; spectrograms, overlays, and **`Clipboard_tutorial_bokeh.ipynb`** remain per **`docs/plans/bokeh-swarm-clipboard-plan.md`** Phases 2–4.
