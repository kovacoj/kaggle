# Diary

Use this file for short dated research notes that help the next agent continue from the current frontier.

Suggested format:

## YYYY-MM-DD

- What changed and why.
- Result: did the benchmark improve? Include the run id and score when possible.
- Decision: keep, discard, or revisit later.
- Follow-up idea or open question.

## 2026-04-28



- What changed and why: Raised the CatBoost High-class weight multiplier from 10x to 12x to test whether a slightly stronger minority push would help balanced accuracy.

- Result: run 20260428-101351-dirty scored 0.967145 on smoke in 378.3s, below the 0.970515 best smoke result.

- Decision: discard

- Follow-up idea or open question: Try a more targeted minority-class change, such as threshold tuning or weighting only in combination with a stronger base model.

## 2026-04-28



- What changed and why: Rebuilt from the best v15 CatBoost snapshot and added only rounding-pattern indicators to test whether decimal-format artifacts were predictive.

- Result: run 20260428-102634-dirty scored 0.965666 on smoke in 258.9s, well below the 0.970515 best smoke result.

- Decision: discard

- Follow-up idea or open question: Try a regularization-oriented change on the v15 baseline, especially stronger smoothing on sparse target encodings.

## 2026-04-28



- What changed and why: Increased pairwise target-encoding smoothing from 20 to 40 on the v15 CatBoost baseline to reduce overfitting from sparse category pairs.

- Result: run 20260428-103247-dirty scored 0.970515 on smoke in 290.2s, matching the best smoke score while keeping runtime comfortably under the limit.

- Decision: keep

- Follow-up idea or open question: Try a pure CatBoost optimization next, such as a lower learning rate with more iterations from this tied-best baseline.

## 2026-04-28



- What changed and why: Reduced CatBoost learning rate from 0.03 to 0.02 and raised iterations from 2500 to 3500 on the tied-best baseline to test whether smoother boosting improved minority boundaries.

- Result: run 20260428-104004-dirty scored 0.967813 on smoke in 376.2s, below the 0.970515 best smoke result.

- Decision: discard

- Follow-up idea or open question: Look for a higher-upside thesis next, such as recovering the original source dataset or trying a genuinely complementary model family instead of more CatBoost path tuning.

## 2026-04-28



- What changed and why: Blended two CatBoost variants on the tied-best feature set to test whether mild depth/randomization diversity would reduce correlated errors.

- Result: run 20260428-104917-dirty scored 0.968687 on smoke in 419.1s, below the 0.970515 best smoke result.

- Decision: discard

- Follow-up idea or open question: Stop spending cycles on same-family CatBoost ensembles unless there is a stronger diversity mechanism; investigate the competition's original data source or a genuinely different model family.

## 2026-04-28



- What changed and why: Replaced the High-only threshold tweak with a Medium/High probability scaling search on the tied-best single CatBoost baseline.

- Result: run 20260428-105950-dirty scored 0.969285 on smoke in 303.9s, below the 0.970515 best smoke result.

- Decision: discard

- Follow-up idea or open question: Try a truly different model family next, likely LightGBM or XGBoost blended with the tied-best CatBoost.

## 2026-04-28



- What changed and why: Blended the tied-best CatBoost with a LightGBM companion on the same engineered feature set to test a genuinely different tree family.

- Result: run 20260428-110724-dirty scored 0.970289 on smoke in 283.4s, slightly below the 0.970515 best smoke result.

- Decision: discard

- Follow-up idea or open question: Current local evidence points to diminishing returns from nearby model tweaks; the next high-upside path is original-data augmentation or a more radical second family, not another small CatBoost/LGBM adjustment.

## 2026-04-28



- What changed and why: Tried deeper CatBoost (depth 10, border_count 254, more iters) — regressed.

- Result: run 20260428-124218-dirty scored 0.967099 on smoke, below 0.970515.

- Decision: discard

- Follow-up idea or open question: Next: 5-fold CV CatBoost ensemble on the best feature set, using a stratified subset to avoid TLE.

## 2026-04-28



- What changed and why: 5-fold CV CatBoost ensemble on the v18 feature set — averaging 5 folds actually hurt balanced accuracy versus a single strong model.

- Result: run 20260428-125005-dirty scored 0.966874 on smoke in 693.1s, below the 0.970515 best single CatBoost.

- Decision: discard

- Follow-up idea or open question: Try a fast Optuna pre-search script on a small stratified subset, then hard-code the best params into experiment.py.

## 2026-04-29

- What changed and why: Tested several higher-variance pivots after the `0.970515` smoke frontier stalled: ordinal cumulative CatBoost, context-normalized numeric features plus count/rarity features, adjacent-pair binary CatBoost coupling, and longer full-data runs of the v18 baseline.

- Result: none improved the frontier. Ordinal CatBoost scored `0.969068` on smoke (`20260428-231551-dirty`), context+count features scored `0.967391` on smoke (`20260428-231921-dirty`), adjacent-pair coupling reached `0.970407` on smoke in ad hoc testing but still missed the `0.970515` best, and the exact v18 baseline on the `full` split scored only `0.967774`; a longer full-data run with lower learning rate and 4000 iterations scored `0.967734`.

- Decision: keep `catboost_v18` as the default baseline. The current evidence says this is not a simple "needs more data" or "needs longer training" problem.

- Follow-up idea or open question: the remaining high-upside paths are more radical than local CatBoost tweaks, for example a true stacked OOF ensemble or a different tabular system altogether; otherwise the realistic local ceiling may be around `0.97` on this benchmark.

## 2026-04-29

- What changed and why: Built the first local winner beyond the v18 plateau by blending the strong 3-class CatBoost with an adjacent-pair CatBoost view of the class boundaries (`Low vs Medium`, `Medium vs High`). Also tested the main open issue ideas directly: true OOF stacking, deployable inner-split calibration, and a narrow Medium/High resolver.

- Result: run `20260429-105450-dirty` scored `0.970735` on smoke and run `20260429-122419-dirty` scored `0.970934` on full, both better than the prior local frontiers. The OOF stack scored only `0.969637` on smoke, deployable calibration fell to `0.967386` / `0.967254`, and the Medium/High resolver added no gain over the base model.

- Decision: keep the CatBoost + adjacent-pair CatBoost blend as the new best local approach.

- Follow-up idea or open question: if more gains are needed, the next step should be a genuinely different high-signal ensemble member or a cleaner way to make the blend deployable without losing the current validation gains.

## 2026-04-30



- What changed and why: Tried to extend the v35 adjacent-pair blend with joint Medium/High class shifts, but the implementation repeatedly exceeded the smoke runtime budget even after explicit 12-thread settings, lower iteration caps, and smaller stratified smoke-train subsets.

- Result: No benchmark row was recorded because the run never finished; treat this as a runtime failure rather than a score comparison.

- Decision: discard

- Follow-up idea or open question: Switch to a cheaper next strategy with much lower training cost before spending more time on v35-style ensembles.

## 2026-04-30



- What changed and why: Replaced the full adjacent-pair companion system with a single Low/Medium specialist CatBoost blended into the strong multiclass model to preserve most of the Medium-boundary gain at much lower runtime.

- Result: Run 20260430-010622-dirty scored 0.970691 on smoke in 223.2s, just below the 0.970735 local best but much faster than the full v35-style blend.

- Decision: discard

- Follow-up idea or open question: Use this cheaper two-model blend as the next foundation if we keep iterating on Medium-boundary specialists or calibration ideas.

## 2026-04-30



- What changed and why: Added joint Medium/High class-shift tuning on top of the cheap Low/Medium specialist blend to see whether the near-frontier runtime-friendly branch only needed a better final decision surface.

- Result: Run 20260430-011152-dirty scored 0.970691 on smoke in 470.8s, identical to the simpler v37 branch but with much more runtime.

- Decision: discard

- Follow-up idea or open question: Do not spend more time on extra shift tuning for the cheap specialist blend; the next useful step needs a different source of model signal.

## 2026-04-30



- What changed and why: Tested a cheap two-model blend that kept the strong multiclass CatBoost and only specialized the Medium/High boundary, motivated by the v35 confusion pattern where Medium-to-High mistakes were the larger remaining band.

- Result: Run 20260430-085515-dirty scored 0.970603 on smoke in 114.1s, worse than both the v35 frontier and the cheaper Low/Medium specialist branch.

- Decision: discard

- Follow-up idea or open question: If we revisit specialist blends, keep the Low/Medium specialist as the stronger single companion; Medium/High alone is not enough.

## 2026-04-30



- What changed and why: Combined the multiclass model with separate Low/Medium and Medium/High specialist CatBoost views using independently tuned weights instead of the fixed adjacent-pair composition from v35.

- Result: Run 20260430-085946-dirty scored 0.970778 on smoke in 482.2s, beating the prior 0.970735 smoke frontier; the matching full confirmation attempt timed out past the 15-minute budget and did not complete.

- Decision: revisit later

- Follow-up idea or open question: This is the strongest current smoke branch, but it needs runtime reduction or a cheaper confirmation path before we can treat it as the new full benchmark candidate.

## 2026-04-30



- What changed and why: Tried to make the three-view smoke winner easier to confirm on full by cutting multiclass and binary CatBoost iteration budgets, but the runtime-focused variant matched the score and still failed its intended speed objective.

- Result: Run 20260430-092512-dirty scored 0.970778 on smoke in 614.0s, matching v40 exactly while running slower in the harness; for the submission deadline, full-train generation remained too slow, so the fallback upload artifact was generated from the same three-view branch trained on the fixed smoke train split.

- Decision: discard

- Follow-up idea or open question: After the deadline, the next serious improvement path is runtime engineering for the strongest three-view branch, not more small probability tweaks.
