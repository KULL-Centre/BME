#   This is a small script for reweighting molecular simulations using
#   the Bayesian/MaxEnt (BME) approach
#   Copyright (C) 2018 Sandro Bottaro (name . surname @ iit . it)
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License V3 as published by
#   the Free Software Foundation, 
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import print_function
import argparse
import bme_reweight as bme

def parse():

    parser = argparse.ArgumentParser(description='This is a Bayesian/MaxEnt reweighting script')
    parser.add_argument("-o", dest="name",help="output_name",default=None,required=True)
    parser.add_argument("--quiet", dest="quiet",help="Be quiet",default=False,required=False,action='store_true')
    parser.add_argument("--exp", dest="exp_files",help="Experimental datafile (s)",nargs="+",default=None,required=True)
    parser.add_argument("--calc", dest="calc_files",help="Back-calculated datafile (s)",nargs="+",default=None,required=True)
    parser.add_argument("--theta", dest="theta",help="Theta value",default=None,required=True,type=float)
    args = parser.parse_args()

    
    return args

def main():

    args = parse()

    bmea = bme.Reweight(verbose=(not args.quiet))
    assert len(args.exp_files)==len(args.calc_files)
    
    for j in range(len(args.exp_files)):
        bmea.load(args.exp_files[j],args.calc_files[j])
        
    bmea.optimize(theta=args.theta)
    
    
if __name__ == "__main__":
    main()
