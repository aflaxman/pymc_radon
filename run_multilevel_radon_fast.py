import pdb
from numpy import *
from pymc import *

import multilevel_radon_fast

mc = MCMC(multilevel_radon_fast)
mc.sample(100000,50000,10)

from pylab import *
errorbar(multilevel_radon_fast.u,
         multilevel_radon_fast.alpha.stats()['mean'],
         multilevel_radon_fast.alpha.stats()['standard deviation'],
         fmt='.', capsize=0)
xlabel('county-level uranium measure')
ylabel('regression intercept')
title('Reproduction of Figure 2')
