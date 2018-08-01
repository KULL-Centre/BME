## toy_model.py ##
The toy model is a two-dimensional model with three states. The mean, covariance matrix
and population of the true probability distribution and of the prior can be set within the script.
Figure 2 was produced with the script toy/toy_model.py

## examples/example1.py ##
example1.py is a simple reweighting script using as restraints scalar couplings. The agrement with
scalar couplings themselves and NOE distances is calculated after reweighting.

## examples/example2.py ##
example2.py is a reweighting script that uses scalar couplings as restraints. Data are divided in 5
blocks to estimate statistical error. The agrement with
scalar couplings themselves, NOE and uNOE distances is calculated after reweighting.

## examples/example3.py ##
example3.py is identical to example2.py, with the difference that different values of theta
are scanned.

## examples/plot.figure3.py ##
Produces panels A,B,C of figure 3

## examples/plot.figure3D.py ##
Produces panel D of figure 3

