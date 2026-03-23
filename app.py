
import pandas as pd
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


data = {
    'attendance': [90, 80, 60, 40, 50, 70, 20, 92],
    'result': ['Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail', 'Fail','Pass']
}

df = pd.DataFrame(data)

X = df[['attendance']]
y = df['result']


def objective(trial):
    max_depth = trial.suggest_int('max_depth', 1, 8)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    score = cross_val_score(model, X, y, cv=2)
    return score.mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2)


print('Best Parameters:', study.best_params)
print('Best Value:', study.best_value)