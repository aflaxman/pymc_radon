import numpy as np
import pymc
import csv
import radon_inv_wishart
from pylab import hist, show
from pymc import Matplot
from timeit import Timer

def run(size='small'):
    M = pymc.MCMC(radon_inv_wishart.model())

    if size == 'small':
        M.sample(iter=1e3, burn=500, thin=5)

    elif size == 'medium':
        M.sample(iter=5e3, burn=1e3, thin=10)

    elif size == 'large':
        M.sample(iter=10e3, burn=5e3, thin=10)

    else:
        raise Error, 'unrecognized size: %s' % size

    fit = M.stats()
    print('mu',fit['mu']['mean'])
    print('xi',fit['xi']['mean'])
    print('sigma_y',fit['sigma_y']['mean'])
    print('tau_y',fit['tau_y']['mean'])

    B = M.trace('B')[:]
    B = sum(B)/len(B)
    outf = open('radon.coefs.from.pymc.csv','w')
    coefsWriter = csv.writer(outf)
    coefsWriter.writerows(B)
    outf.close()
