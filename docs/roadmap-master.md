# Roadmap Master: All Improvement Roadmaps

> All roadmaps organized by priority with dependency graph and implementation order.

---

## Priority Matrix

| Priority | Roadmap | Effort | Deps | Key Benefit |
|----------|---------|--------|------|-------------|
| **P0** | [Data Leakage Prevention](roadmap-data-leakage-prevention.md) | Med | None | Correct training data integrity |
| **P0** | [Difference Features & Debiasing](roadmap-difference-features-and-debiasing.md) | Med | None | Core feature upgrade (11→45+ features) |
| **P0** | [Temporal CV & Tuning](roadmap-temporal-cv-and-tuning.md) | Med | None | Honest evaluation, Optuna hyperparameters |
| **P1** | [Calibration & Evaluation](roadmap-calibration-and-evaluation.md) | Low | P0 temporal CV | Calibrated probabilities, proper metrics |
| **P1** | [Matchup Interaction Features](roadmap-matchup-interaction-features.md) | Med | P0 diff features | 11 additional matchup features |
| **P1** | [Forward Simulation Engine](roadmap-forward-simulation-engine.md) | Med | P1 calibration | Full bracket, survivor pool, upset tracking |
| **P1** | [Model Config & Versioning](roadmap-model-config-and-versioning.md) | Low | None | Reproducibility, experiment tracking |
| **P1** | [Project Architecture](roadmap-project-architecture.md) | High | None | Code organization, maintainability |
| **P2** | [Stacking Meta-Learner](roadmap-stacking-meta-learner.md) | Med | P0 temporal CV, P0 diff features | Ensemble upgrade from averaging to learned stacking |
| **P2** | [Game Data & Advanced Features](roadmap-game-data-and-advanced-features.md) | High | P0 diff features | Elo, four factors, momentum, composite ratings |
| **P2** | [Sample Weighting & Training Strategy](roadmap-sample-weighting-and-training-strategy.md) | Low | P0 temporal CV | Better weighting logic, all-50 ensemble, seeds |
| **P2** | [Testing & CI](roadmap-testing-and-ci.md) | Med | P1 architecture | Automated testing, regression checks |
| **P2** | [LightGBM & Model Exploration](roadmap-lightgbm-and-model-exploration.md) | Low | P2 stacking | Alternative model types for ensemble diversity |
| **P3** | [Live Prediction Pipeline](roadmap-live-prediction-pipeline.md) | High | P1 calibration, P1 config | Automated live predictions, ROI dashboard |

---

## Dependency Graph

```
                    ┌──────────────────────┐
                    │  P0: Data Leakage    │
                    │      Prevention      │
                    └──────────────────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     │                        │                        │
     ▼                        ▼                        ▼
┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐
│P0: Difference│   │ P0: Temporal CV  │   │ P1: Model Config │
│   Features   │   │    & Tuning      │   │  & Versioning    │
└──────────────┘   └──────────────────┘   └──────────────────┘
     │                   │       │                    │
     ├───────────────────┤       │                    │
     │                   │       │                    │
     ▼                   ▼       ▼                    ▼
┌──────────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐
│P1: Matchup   │  │P1: Calib │  │P2: Stacking  │  │P3: Live      │
│ Interaction  │  │& Eval    │  │Meta-Learner  │  │ Pipeline     │
└──────────────┘  └──────────┘  └──────────────┘  └──────────────┘
                       │              │
                       ▼              ▼
                  ┌──────────┐  ┌──────────────┐
                  │P1: Fwd   │  │P2: LightGBM  │
                  │Simulation│  │ & Model Expl. │
                  └──────────┘  └──────────────┘

  Independent tracks:
  ┌──────────────────┐     ┌──────────────────┐
  │ P1: Project      │────▶│ P2: Testing & CI │
  │   Architecture   │     │                  │
  └──────────────────┘     └──────────────────┘

  ┌──────────────────┐     ┌──────────────────┐
  │ P2: Game Data &  │     │ P2: Sample       │
  │ Advanced Features│     │ Weighting        │
  └──────────────────┘     └──────────────────┘
```

---

## Recommended Implementation Order

### Sprint 1 — Foundation (Do First)
1. **Data Leakage Prevention** — Fix BartTorvik temporal integrity
2. **Difference Features & Debiasing** — Introduce `team_A - team_B` features, symmetrization, column-swap debiasing
3. **Temporal CV & Tuning** — Leave-year-out CV, Optuna hyperparameter search
4. **Model Config & Versioning** — Centralized YAML config for reproducibility

### Sprint 2 — Calibration & Evaluation
5. **Calibration & Evaluation** — Isotonic regression, log loss, Brier score, reliability diagrams
6. **Matchup Interaction Features** — Tempo pace mismatch, size mismatches, style clashes

### Sprint 3 — Ensemble & Simulation
7. **Stacking Meta-Learner** — Out-of-fold predictions, learned stacking weights
8. **Forward Simulation Engine** — Deterministic + Monte Carlo bracket simulation

### Sprint 4 — Polish & Expand
9. **Project Architecture** — Reorganize into clean package structure
10. **Game Data & Advanced Features** — Elo, four factors, momentum
11. **Sample Weighting & Training Strategy** — Refined weighting, all-50 ensemble
12. **LightGBM & Model Exploration** — LightGBM, CatBoost, MLP as ensemble members

### Sprint 5 — Automation
13. **Testing & CI** — pytest, GitHub Actions, regression testing
14. **Live Prediction Pipeline** — Automated pipeline, ROI dashboard

---

## Existing Reference Documentation

These analysis docs (already in `docs/`) provide background research:

| File | Content |
|------|---------|
| [sports-quant-integration-master.md](sports-quant-integration-master.md) | Master gap analysis table |
| [sports-quant-features-models.md](sports-quant-features-models.md) | Feature & model comparison |
| [sports-quant-calibration-tuning.md](sports-quant-calibration-tuning.md) | Calibration & tuning approaches |
| [sports-quant-scraping-data-sources.md](sports-quant-scraping-data-sources.md) | Data pipeline comparison |
| [sports-quant-simulation-survivor.md](sports-quant-simulation-survivor.md) | Bracket simulation details |
| [sports-quant-injury-adjustment.md](sports-quant-injury-adjustment.md) | Injury adjustment system |
