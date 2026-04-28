# Ideas for Next Experiments

Agents: append your ideas here. Mark them [tried] once tested.

- Stacking: use CatBoost/LightGBM/XGBoost predictions as features into a logistic regression or ridge meta-learner (instead of simple averaging).
- Optuna hyperparameter search on the best single model (catboost_v12) — focus on depth, learning_rate, l2_leaf_reg, bagging_temperature.
- Original dataset: the competition mentions a source dataset. Download it and concatenate with the synthetic train data (with a `is_original` flag feature).
- More aggressive High-class weighting: current best uses logit/digit thresholds. Try SMOTE oversampling for High class instead.
- K-fold cross-validation: train with 5-fold CV on `full` benchmark, average predictions. Reduces variance.
- Feature selection: use SHAP importance from catboost_v12 to drop noisy features and retrain.
- Neural network: a small tabular MLP (2-3 hidden layers) as an additional ensemble member.
