"""Class 11, Exercise A: Probability Grammatics and `ProGED`"""

import os, sys
sys.path.append(os.path.join(os.getcwd(), os.pardir, "10"))  # enable importing data_generators from 'ex/10/'

import numpy as np
import ProGED as pg

import data_generators
from aml import fancy_print




# 1 Discover Newton's Second Law
fancy_print("NEWTON'S SECOND LAW")
DATA_GENERATOR_1 = data_generators.generate_newton
data_1 = DATA_GENERATOR_1(100)
np.random.seed(1)

ED_1 = pg.EqDisco(data=data_1,
                lhs_vars=["F"],
                rhs_vars=["m", "a"],
                sample_size=10)

ED_1.generate_models()
ED_1.fit_models()
print(ED_1.get_results())
del data_1, ED_1


# 2 Discover a linear function
fancy_print("LINEAR FUNCTION")
DATA_GENERATOR_2 = data_generators.generate_linear
data_2 = DATA_GENERATOR_2(100)
np.random.seed(1)

grammar = "E -> E '+' V [0.6] | V [0.4]\n"
grammar += "V -> 'x1' [0.33] | 'x2' [0.67]"
Grammar = pg.generators.GeneratorGrammar(grammar)

ED_2 = pg.EqDisco(data=data_2,
                lhs_vars=["y"],
                rhs_vars=["x1", "x2"],
                sample_size=15,
                generator=Grammar)

ED_2.generate_models()
ED_2.fit_models()
print(ED_2.get_results())
del data_2,


# 3 Discover the energy conservation law
