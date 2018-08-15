#!/bin/bash

# run this script using bash do_figures.sh to produce figures 2 and 3 in manuscript
set -e
cd ../toy_model/
# make fig 2A,2C,2D,2E
python toy_model.py
mv fig02* ../figures/

# make fig2D
cd ../figures/
python plot.figure2B.py

# produce data for figure 3
cd ../examples/
python example2.py
python example3.py

cd ../figures/
python plot.figure3.py
python plot.figure3D.py
