import csv
from numpy import *
from pymc import *

household_data = [d for d in csv.DictReader(open('household.csv'))]
county_data = [d for d in csv.DictReader(open('county.csv'))]

# hyper-priors
g = Uniform('gamma', [0,0], [100,100])

s_a = Uniform('sigma_a', 0, 100)

# priors
a = {}
for d in county_data:
    @stochastic(name='a_%s'%d['county'])
    def a_j(value=0., g=g, u_j=float(d['u']), s_a=s_a):
        return normal_like(value, g[0] + g[1]*u_j, s_a**-2.)
    a[d['county']] = a_j

b = Uniform('beta', 0, 100)

s_y = Uniform('sigma_y', 0, 100)

# likelihood
y = {}
for d in household_data:
    @stochastic(observed=True, name='y_%s'%d['household'])
    def y_i(value=float(d['y']), a_j=a[d['county']], b=b,
            x_ij=float(d['x']), s_y=s_y):
        return normal_like(value, a_j + b*x_ij, s_y**-2.)
    y[d['household']] = y_i

mc = MCMC([g, s_a, a, b, s_y, y])
mc.sample(1000)

from pylab import *
errorbar([d['u'] for d in county_data],
          [a[d['county']].stats()['mean'] for d in county_data],
          [a[d['county']].stats()['standard deviation'] for d in county_data],
          fmt='.', capsize=0)
xlabel('county-level uranium measure')
ylabel('regression intercept')
title('Reproduction of Figure 2')
