# Branch Summary: feature/plotly-interactive-views

## Metadata
- **Branch**: feature/plotly-interactive-views
- **Status**: open
- **Opened on**: 2026-04-18
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
P26-04-18

## Accomplishments
- Created branch `feature/plotly-interactive-views` for Plotly-backed Map, CrossSection, and Clipboard work.
- Documented mandatory branch policy and aligned copy-paste prompts in `docs/plotly_map_crosssection_clipboard_plan.md`.
- Added optional `[plotly]` extra and a matplotlib-free cross-section data layer under `vdapseisutils.core.maps.plotly` (Part 1 foundation).

## Planned work
- CrossSection → Map → Clipboard Plotly ports per `docs/plotly_map_crosssection_clipboard_plan.md` (nine phased prompts).
- CrossSection Plotly Part 2: interactive `go.Figure` wired to the Part 1 data structures.

## Executed work
- Branch created; plan committed with branch-only policy and prompt updates.
- Part 1: `pyproject` `[plotly]` extra, `cross_section_data` + `empty_cross_section_figure`, README note, tests.

## Back-and-forth / iteration notes
- 

## Problems + resolutions
- 

## Validation
- `pytest tests/test_cross_section_plotly_data.py tests/test_maps_stack_smoke.py` (local).

## Final changelog-style outcome
- 
