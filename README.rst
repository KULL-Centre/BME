.. image:: https://travis-ci.org/KULL-Centre/BME.svg
	       :target: https://travis-ci.org/KULL-Centre/BME

			
Integrating Molecular Simulation and Experimental Data:  A Bayesian/Maximum Entropy Approach
==============

This is a Python script to perform ensemble refinement using the Bayesian/MaxEnt (BME) approach.
You may want to use this code when you have a molecular simulation for which calculated averages do not match available experimental data (eg chemical shifts, NOE, scalar couplings, SAXS measurements, etc.). In this case, you can use the experimental data to perform an a posteriori correction of your simulation.
The correction comes in the form of a new set of weights, one per frame in your simulation, so that calculated averages match the experimental data within some uncertainty. For a detailed description of the algorithm see our manuscript

::

   @article{bottaro2018integrating,
   title={Integrating Molecular Simulation and Experimental Data: A Bayesian/Maximum Entropy Reweighting Approach},
   author={Bottaro, Sandro and Bengtsen, Tone and Lindorff-Larsen, Kresten},
   journal={bioRxiv},
   pages={457952},
   year={2018},
   publisher={Cold Spring Harbor Laboratory}
   }


Requirements 
------------

1) Python>=2.7 or Python>=3.3

2) Numpy, Scipy libraries

3) Jupyter and Matplotlib (for notebooks only)
  
  
Download 
-----------

You can download a .zip file by clicking on the green button above or using git

`git clone https://github.com/sbottaro/BME.git`


Examples
------------

Simple examples can be found in the `examples` folder. The `notebook` folder contains more detailed examples in form of jupyter notebooks. 
Note also that the software can be used from the commandline:

   python bme.py --exp exp_datafile --calc calc_datafile -o outfile --theta 1.0

Contacts, references and other stuff
--------------

For further questions, send an email to sandro_dot_bottaro(guesswhat)dot_iit_dot_it
You may consider reading and citing the following relevant references as well:

    
::
   
    @article{rozycki2011saxs,
    title={SAXS ensemble refinement of ESCRT-III CHMP3 conformational transitions},
    author={R{\'o}{\.z}ycki, Bartosz and Kim, Young C and Hummer, Gerhard},
    journal={Structure},
    volume={19},
    number={1},
    pages={109--116},
    year={2011},
    publisher={Elsevier}
    }
    
::

   @article{bottaro2018conformational,
    title={Conformational ensembles of RNA oligonucleotides from integrating NMR and molecular simulations},
    author={Bottaro, Sandro and Bussi, Giovanni and Kennedy, Scott D and Turner, Douglas H and Lindorff-Larsen, Kresten},
    journal={Science Advances},
    volume={4},
    number={5},
    pages={eaar8521},
    year={2018},
    publisher={American Association for the Advancement of Science}
    }		

    
::

    @article{hummer2015bayesian,
    title={Bayesian ensemble refinement by replica simulations and reweighting},
    author={Hummer, Gerhard and K{\"o}finger, J{\"u}rgen},
    journal={The Journal of chemical physics},
    volume={143},
    number={24},
    pages={12B634\_1},
    year={2015},
    publisher={AIP Publishing}
    }

::

    @article{cesari2016combining,
    title={Combining simulations and solution experiments as a paradigm for RNA force field refinement},
    author={Cesari, Andrea and Gil-Ley, Alejandro and Bussi, Giovanni},
    journal={Journal of chemical theory and computation},
    volume={12},
    number={12},
    pages={6192--6200},
    year={2016},
    publisher={ACS Publications}
    }


::

    @article{cesari2018using,
    title={Using the maximum entropy principle to combine simulations and solution experiments},
    author={Cesari, Andrea and Rei{\ss}er, Sabine and Bussi, Giovanni},
    journal={Computation},
    volume={6},
    number={1},
    pages={15},
    year={2018},
    publisher={Multidisciplinary Digital Publishing Institute}
    }
		


