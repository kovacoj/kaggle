# Ideas for Next Experiments

Agents: append your ideas here before coding. Mark them `[tried]` when execution starts and add a short outcome note after the run.

Suggested entry format:

- `[new]` Idea title | hypothesis: ... | expect: ... | note: pending
- `[tried]` Idea title | hypothesis: ... | expect: ... | note: improved smoke by ..., crashed because ..., or no signal

- `[new]` Stacking | hypothesis: use CatBoost/LightGBM/XGBoost predictions as features into a logistic regression or ridge meta-learner instead of simple averaging | expect: better calibration and balanced accuracy than naive blending | note: pending
- `[new]` Optuna on best single CatBoost | hypothesis: the current CatBoost feature set is stronger than the ensemble, so a focused search over depth, learning_rate, l2_leaf_reg, and bagging_temperature may beat the current best | expect: small but real smoke improvement without extra model complexity | note: pending
- `[new]` Original dataset augmentation | hypothesis: concatenate the source irrigation dataset with the synthetic train data and add an `is_original` flag feature | expect: more robust minority-class signal, especially for `High` | note: pending
- `[new]` Stronger High-class weighting | hypothesis: the best current setup still under-serves `High`, so more aggressive weighting or oversampling may lift balanced accuracy | expect: recall gain on `High` without collapsing `Low` recall | note: pending
- `[new]` K-fold CV ensemble | hypothesis: averaging 5 folds on the `full` benchmark will reduce variance versus a single split fit | expect: stabler validation score and possibly stronger submission performance | note: pending
- `[new]` Feature selection with SHAP | hypothesis: dropping noisy features from the best CatBoost run will simplify the model without losing signal | expect: similar or better score with less runtime and lower overfit risk | note: pending
- `[new]` Tabular MLP ensemble member | hypothesis: a small neural net may capture interactions the tree models miss | expect: complementary errors that improve an ensemble | note: pending
