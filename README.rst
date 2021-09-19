
.. image:: https://app.travis-ci.com/KULL-Centre/BME.svg?branch=main
    :target: https://app.travis-ci.com/KULL-Centre/BME
    
Integrating Molecular Simulation and Experimental Data:  A Bayesian/Maximum Entropy Approach
==============

This is a Python script to perform ensemble reweighting using the Bayesian/MaxEnt (BME) approach.
You may want to use this code when you have a molecular simulation for which calculated averages do not match available experimental data (eg chemical shifts, NOE, scalar couplings, SAXS, etc.). In this case, you can use the experimental data to perform an a posteriori correction of your simulation.
The correction comes in the form of a new set of weights, one per frame in your simulation, so that calculated averages match the experimental data within some uncertainty. For a detailed description of the algorithm see our manuscript here_

::

	@incollection{bottaro2020integrating,
	title={Integrating molecular simulation and experimental data: a Bayesian/maximum entropy reweighting approach},
  	author={Bottaro, Sandro and Bengtsen, Tone and Lindorff-Larsen, Kresten},
  	booktitle={Structural Bioinformatics},
  	pages={219--240},
  	year={2020},
  	publisher={Springer}
	}



Requirements 
------------

1) Python>=3.4

2) Numpy, Scipy, Sklearn, Pandas

3) Jupyter and Matplotlib (for notebooks only)
  
  
Download 
-----------

You can download a .zip file by clicking on the green button above or using git

`git clone https://github.com/sbottaro/BME.git`


Examples
------------

The `notebook` folder contains more detailed examples in form of jupyter notebooks. 

- Example_01: Introduction
- Example_02: How to chose the regularization parameter $\theta$
- Example_03: How to handle different types of data simultaneously
- Example_04: Iterative BME (iBME)
- Example_05: How to set non-uniform initial weights


BME has been used in several integrative studies, including:

- Computing, Analyzing, and Comparing the Radius of Gyration and Hydrodynamic Radius in Conformational Ensembles of Intrinsically Disordered Proteins (`Preprint <https://www.biorxiv.org/content/10.1101/679373v2>`_) (`CODE <https://github.com/KULL-Centre/papers/tree/master/2019/IDP-methods-Ahmed-et-al>`_)
- Integrating NMR and Simulations Reveals Motions in the UUCG Tetraloop (`Article <https://academic.oup.com/nar/article/48/11/5839/5840580>`_) (`CODE <https://github.com/KULL-Centre/papers/edit/master/2020/UUCG-dynamics-Bottaro-et-al/README>`_)
- Structure and dynamics of a nanodisc by integrating NMR, SAXS and SANS experiments with molecular dynamics simulations (`Article <https://elifesciences.org/articles/56518>`_) (`CODE <https://github.com/KULL-Centre/papers/tree/master/2020/nanodisc-bengtsen-et-al>`_)
- Bayesian-Maximum-Entropy Reweighting of IDP Ensembles Based on NMR Chemical Shifts (`Article <https://www.mdpi.com/1099-4300/21/9/898>`_)
- Side chain to main chain hydrogen bonds stabilize a polyglutamine helix in a transcription factor (`Article https://www.nature.com/articles/s41467-019-09923-2`_)
- Interplay of folded domains and the disordered low-complexity domain in mediating hnRNPA1 phase separation (`Article <https://academic.oup.com/nar/article/49/5/2931/6134187>`_) (`CODE <https://github.com/KULL-Centre/papers/tree/master/2020/hnRNPA1-martin-et-al>`_)
-  Combining molecular dynamics simulations with small-angle X-ray and neutron scattering data to study multi-domain proteins in solution (`Article <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007870>`_) (`CODE <https://github.com/KULL-Centre/papers/tree/master/2020/TIA1-SAS-Larsen-et-al>`_)
- Architecture and assembly dynamics of the essential mitochondrial chaperone complex TIM9·10·12 (`ARTICLE <https://www.sciencedirect.com/science/article/pii/S0969212621001258>`_)
- Properdin oligomers adopt rigid extended conformations supporting function (`ARTICLE <https://elifesciences.org/articles/63356>`_)
- Refinement of alpha-Synuclein Ensembles Against SAXS Data: Comparison of Force Fields and Methods (`ARTICLE <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8100456/>`_)
- Structural basis of client specificity in mitochondrial membrane-protein chaperones (`ARTICLE <https://advances.sciencemag.org/content/6/51/eabd0263>`_)
- Order and disorder – an integrative structure of the full-length human growth hormone receptor (`ARTICLE <https://www.biorxiv.org/content/10.1101/2020.06.25.171116v1.abstract>`_)
- Structural Basis of Membrane Protein Chaperoning through the Mitochondrial Intermembrane Space (`ARTICLE <https://www.sciencedirect.com/science/article/pii/S0092867418313953>`_)
   
   
Contacts, references and other stuff
--------------

For further questions, send an email to sandro_dot_bottaro(guesswhat)dot_bio_dot_ku_dot_dk
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
		

.. _here: https://www.biorxiv.org/content/10.1101/457952v1
