# Case Construction Report: astra_1688_trend_trader_001

## Overview

This document records the full lifecycle of constructing `astra_1688_trend_trader_001`, the first hard fixed case migrated from Astra to Bella. It covers source selection, demand design, verify construction, and iterative refinement based on multi-model evaluation.

## Source Selection

### Selection Criteria

Three signals were combined to identify hard cases from the Astra lite-e1 corpus (500 environments, 15000 tasks):

1. **Low environment acceptance rate** — environments where most trajectories failed eval
2. **Long accepted trajectories** — cases that passed but required many tool calls
3. **Low probe scores** — cases where small models scored poorly at all depths

### Why `09432_1688-trend-trader`

| Metric | Value |
|--------|-------|
| Environment acceptance rate (GPT-5.4 eval) | 30% (9/30 accepted) |
| Probe score (avg across depths) | 0.519 |
| Combined hardness rank | #1 out of 500 environments |
| Tools | 29 (13 mutation, 16 query) |
| DB tables | 11 |
| DB rows | 179 |

### Why task_020

| Metric | Value |
|--------|-------|
| Eval verdict | Rejected by GPT-5.4, accepted on Opus 4.6 recall |
| Task rule | miss_param (user omits details, agent must discover) |
| Trajectory length | 62 messages, 15 user turns, 15 tool calls |
| Workflow complexity | Non-linear (create → list → delete → recreate) |

## Environment Migration

Copied from `/Data/bywei/tmp/astra-environments-lite-e1/09432_1688-trend-trader/` to `environments/09432_1688-trend-trader/`. Structure: `contract/` (ENVIRONMENT.md, tools.jsonl, tool_graph.json), `runtime/` (backend.py), `world/` (world.db, schema.sql).

No modifications to the environment itself.

## Demand Design

### Original Astra demand (dynamic mode)

```
I'm from HarvestLoop Commerce and I need to get logged in and set up a new sourcing
project for baby feeding products — we're looking at selling them to European shoppers
through one of our online storefronts. Add that kids tableware factory from Qingdao to
the project shortlist and confirm it was added correctly. Check their compliance
certifications to make sure everything is in order. Run the margin projections so I can
see whether the economics make sense. Draft an initial inquiry to the factory about
sample availability, then I'll want to see what inquiries are on file, delete that first
draft, and rewrite a firmer version focused on OEM packaging customization instead. Once
everything is gathered, create a sourcing report covering the supplier evaluation and
projected margins, then update the report with finalized next steps. I also want to
review my current projects and reports to keep track of everything.
```

### Adaptations for fixed mode

1. **Added credentials** — original was `miss_param` rule where user provides credentials when asked. Fixed mode agent cannot ask, so credentials are included directly in the demand.

2. **Removed misleading market phrasing** — original said "targeting European shoppers" and "in order for Europe", which misled models into filtering `search_factory_offers` by `target_market="europe"`. However, the correct offer (OF0019) has `recommended_market="north_america"` in the database. Changed to neutral phrasing that specifies Europe as the project target without implying it should be used as a search filter on offers.

3. **Added search hint** — changed "look it up and add it" to "find their offer for the baby feeding set" to guide toward `product_id`-based search rather than keyword-only approaches.

### Final demand (v2)

```
I'm Amelia Stone from HarvestLoop Commerce (username: amelia.growth, password:
HarvestLoop!26). Log me in and set up a new sourcing project for a baby feeding set —
target market Europe, sell on Shopify at 21 EUR, 600 units per month. There's a kids
tableware factory in Qingdao that offers this product — find their offer for the baby
feeding set, add it to the project shortlist, and confirm the entry is correct. Check
the factory's compliance certifications. Run the margin projections for Shopify at that
sell price so I can see whether the economics work. Draft an initial inquiry to the
factory about sample availability, then show me what inquiries are on file, delete that
first draft, and rewrite a firmer version focused on OEM packaging customization instead.
Once that's done, create a sourcing report covering the supplier evaluation and projected
margins, then update the report with finalized next steps. Finally, pull up my current
projects and reports so I can see everything in one view.
```

## Expected Tool Call Chain (~15 steps)

1. `authenticate_buyer(username, password)` → session token
2. `search_products(keyword="baby feeding")` → PR0010
3. `create_sourcing_project(product_id=PR0010, target_market=europe, sales_channel=shopify, sell_price=21, units=600)`→ PJ0015
4. `search_factory_offers(product_id="PR0010")` → finds OF0019 (factory FC0010 in Qingdao)
5. `add_project_factory(project_id=PJ0015, offer_id=OF0019)` → PF0021
6. `get_project_factory(PF0021)` — confirm shortlist
7. `list_factory_certificates(factory_id=FC0010)` — compliance check
8. `calculate_profit_projection(offer_id=OF0019, sell_price=21, units=600, channel=shopify)`
9. `create_inquiry_draft(type=sample, tone=collaborative)` → IN0018
10. `list_inquiry_drafts(project_id=PJ0015)` — show on-file inquiries
11. `delete_inquiry_draft(IN0018)`
12. `create_inquiry_draft(type=oem, tone=firm)` → IN0018 (ID reused)
13. `create_sourcing_report(project_id=PJ0015)` → RP0014
14. `update_sourcing_report(report_id=RP0014, next_action=...)`
15. `list_sourcing_projects` + `list_sourcing_reports` — final overview

## Verify SQL Design

### Principles

- Filter by business keys (`buyer_id`, `product_id`, `offer_id`), not auto-generated IDs
- Never verify non-deterministic fields (timestamps, tokens, generated text)
- Each verify checks one semantic outcome independently

### Verify specifications

| # | What it checks | Key insight |
|---|----------------|-------------|
| 0 | Project created with correct params | sell_price=21.0, units=600, market=europe, channel=shopify |
| 1 | Shortlist entry exists | OF0019 linked to the project (status not checked) |
| 2 | Sample inquiry was deleted | COUNT=0 for sample type on this project |
| 3 | OEM inquiry exists with firm tone | inquiry_type=oem, tone=firm |
| 4 | Report exists | COUNT=1 for this project |
| 5 | Report has non-empty next_action | length(next_action) > 0 |

### Verify iteration

Original verify[1] checked `shortlist_status = 'watchlist'`. This was too strict — a reasonable agent might advance the status to `contacting` after adding the factory. Changed to only check existence.

## Evaluation Results

### v1 (before demand fix)

| Model | pass@1 | Notes |
|-------|--------|-------|
| GPT-5.2 | 0/8 (0%) | All stuck on `target_market="europe"` filter |
| Claude Opus 4.6 | 7/8 (87.5%) | 1 failure due to overly strict verify |

### v2 (after demand fix + verify relaxation)

| Model | pass@1 | Notes |
|-------|--------|-------|
| GPT-5.2 | 1/8 (12.5%) | Still systematic failure on search strategy |
| Claude Opus 4.6 | 8/8 (100%) | All pass |

## Failure Analysis

### GPT-5.2 systematic failure mode

Across 15 failed trials (v1 + v2 combined), GPT-5.2 exhibits a consistent pattern:

1. **`target_market` fixation** — every `search_factory_offers` call includes `target_market="europe"`. The model never drops this parameter despite repeated 0-result responses. OF0019's `recommended_market` is actually `north_america`, so this filter always excludes the correct result.

2. **Keyword-only variation** — when search returns 0, the model varies keywords but never removes or changes other parameters. It treats parameters as invariants rather than hypotheses.

3. **Premature abandonment** — several trials give up after 8-10 calls. Even when finding FC0010 via `search_factories(keyword="Qingdao")`, the model fails to connect it to an offer because it cannot find the offer through `search_factory_offers` with the europe filter.

4. **Skip critical steps** — in trials where it does find the factory, it skips `add_project_factory` and jumps directly to later steps (inquiry, report), causing cascading failures.

### Claude Opus 4.6 success pattern

All 8 v2 passes share a common strategy:

1. **Progressive filter relaxation** — starts with restrictive searches, then systematically removes parameters. Crucially, tries `search_factory_offers(product_id="PR0010")` without `target_market` after the filtered version returns 0.

2. **Multiple search dimensions** — uses varied keywords (`feeding`, `kids`, `baby`, `tableware`, `silicone`) across both `search_factories` and `search_factory_offers`.

3. **Correct dependency chain** — always calls `add_project_factory` before any operation that requires `project_factory_id`.

4. **Complete workflow execution** — all 15 logical steps completed in 18-24 tool calls.

## Data Inconsistency Finding

During analysis, we discovered that OF0019's `recommended_market = "north_america"` while the demand describes a European-focused project. This is an artifact of the Astra environment synthesis — the offer data and task demand were generated independently.

This inconsistency acts as a search trap: models that logically infer "European project → filter offers by europe" will fail. The fix was to adjust demand phrasing to avoid implying offer-level market filtering, while preserving the project-level market specification.

## Lessons Learned

1. **Demand-data consistency matters** — when demand implies a constraint that would exclude the correct answer, it tests search resilience rather than task understanding. This can be valid (testing error recovery) or unfair (data bug). Must be intentional.

2. **Verify should allow reasonable variation** — agents may take sensible additional actions (like advancing shortlist status). Verify should check outcomes, not exact intermediate states.

3. **Single-parameter fixation is a real model differentiator** — GPT-5.2's inability to drop `target_market` after repeated failures is a systematic capability gap, not random noise.

4. **Home advantage is real but bounded** — Opus generated the original trajectory, and the environment was designed in that ecosystem. After fixing the data inconsistency, the gap persists (12.5% vs 100%), confirming genuine capability difference beyond familiarity.
