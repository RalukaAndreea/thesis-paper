import sys
sys.path.append('.')
from automate_feature_selection_extended import load_extended_training_data, SEED
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

X, y = load_extended_training_data()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=4, min_samples_split=2, class_weight='balanced', random_state=SEED, n_jobs=-1)
rfecv = RFECV(estimator=rf, step=1, cv=cv, scoring='f1_weighted', min_features_to_select=1, n_jobs=-1)
rfecv.fit(X, y)

scores = rfecv.cv_results_['mean_test_score'] if hasattr(rfecv, 'cv_results_') else rfecv.grid_scores_
print("SCORES:")
for i, s in enumerate(scores):
    print(f"{i+1} features: {s:.4f}")
