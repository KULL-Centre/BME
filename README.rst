
This is a python2.7 script to reweight trajectories using Bayesian/MaxEnt (BME) restraints from experimental data.
In its simplest form the script can be used from the commandline:

   python bme.py --exp exp_datafile --calc calc_datafile -o outfile --theta 1.0

More complex examples are provided in the examples and in the notebook folders. A manuscript describing in more detail the procedure will be soon available on biorxiv. 
For further questions, send an email to
sandro_dot_bottaro(guesswhat)bio_dot_ku_dot_dk.
If you use this script, please cite the following paper:

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

You should consider reading and citing the following relevant references as well:
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
		


