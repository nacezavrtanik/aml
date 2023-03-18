"""Class 4, Exercise A: Hyperparameter Optimisation with hyperopt"""

from hyperopt import hp, tpe, rand, fmin, Trials, space_eval, pyll, base

from aml import fancy_print


# 1 Minimise function of one variable
space = {'x': hp.uniform('x', 4, 8)}


def objective(params):
    x = params['x']
    return x**2


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

best = space_eval(space, best)
fancy_print('Best x', best.get('x'))
fancy_print('Best value', objective(best))
