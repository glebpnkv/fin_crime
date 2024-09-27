from sklearn.metrics import precision_score, recall_score, f1_score

import xgboost as xgb


def custom_eval_metrics(y_pred, dmatrix):
    y_true = dmatrix.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, average=None)
    recall = recall_score(y_true, y_pred_binary, average=None)

    # Return a list of tuples (metric_name, value)
    out = [
        ('f1', f1),
        ('precision_class_0', precision[0]),
        ('recall_class_0', recall[0]),
        ('precision_class_1', precision[1]),
        ('recall_class_1', recall[1])  # Last metric controls early stopping; fraud recall
    ]

    return out


def fit_xgb_classifier(
        params,
        df,
        x_cols,
        y_col,
        train_idx,
        test_idx,
        num_boost_round=1000,
        early_stopping_rounds=10
):
    x_train = df.loc[train_idx, x_cols]
    y_train = df.loc[train_idx, y_col]

    x_test = df.loc[test_idx, x_cols]
    y_test = df.loc[test_idx, y_col]

    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    cur_evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        evals_result=cur_evals_result,
        early_stopping_rounds=early_stopping_rounds,
        custom_metric=custom_eval_metrics,
        maximize=True,
        verbose_eval=None
    )

    return cur_evals_result, model
