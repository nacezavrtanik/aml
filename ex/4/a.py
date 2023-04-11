"""Class 4, Exercise A: Hyperparameter Optimisation with hyperopt"""

from hyperopt import hp, tpe, rand, fmin, Trials, space_eval, pyll, base

from aml import fancy_print


# 1 Minimise function of one variable, uniform distribution
space_x = {'x': hp.uniform('x', 4, 8)}


def objective(params):
    x = params.get('x')
    return x**2


trials = Trials()
best = fmin(fn=objective,
            space=space_x,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

best = space_eval(space_x, best)
fancy_print('Best x', best.get('x'))
fancy_print('Best value', objective(best))

# 2 Minimise function of two variables, normal distribution
space_xy = {'x': hp.normal('x', -2, 2),
            'y': hp.normal('y', -1, 3)}


def objective(params):
    x = params.get('x')
    y = params.get('y')
    return (x - y)**2 + (y - 1)**2


trials = Trials()
best = fmin(fn=objective,
            space=space_xy,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

best = space_eval(space_xy, best)
fancy_print('Best x', best.get('x'))
fancy_print('Best y', best.get('y'))
fancy_print('Best value', objective(best))
