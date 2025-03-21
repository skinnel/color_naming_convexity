# clmsThesis

## Overview

Repository for all things related to my M.S. thesis, 
\"Convexity is a Fundamental Feature of Efficient Semantic Compression in Probability Spaces.\"

This repository is designed to test the conditions under which IB optimized word mappings yield 
quasi-concave probability functions over the meaning space. 

In order to generate the natural language encoders you must first download the world color survey (WCS) data from 
https://linguistics.berkeley.edu/wcs/data.html. This data should be stored in a single director that contains 
chip.txt, term.txt, lang.txt, and dict.txt from the WCS data. 

The repository is comprised of the following:

chart_creation: 
Contains script necessary to generate the final charts and analyses used in the paper. 
combined_chart_creation.py is the main script to run to create the charts and statistical analyses used in 
the paper, using the pre-computed results in the res directory.

empirical_tests: 
The main scripts of interest from here are 
* generate_suboptimal_encoders: Takes the optimal encoders and performs the shuffling to generate the sub-optimal 
encoders.
* get_convexity_from_encoders: Calculates and saves the quasi-convexity values for specified encoders.
* get_optimal_encoders: Generates the optimal encoders from the pre-computed encoders available from https://github.com/nogazs/ib-color-naming
* natural_languages_assessment: Generates the natural language encoders from the WCS data. Calls functions defined in
wcs_data_processing. 
* utils: Contain the quasi-convexity calculation and helper functions.
* wcs_data_processing: Converts the WCS data into an ingestible format to create the NL encoders.

ib_color_naming: taken from https://github.com/nogazs/ib-color-naming to gain access to assessment metrics for comparison 
and optimal model results.

ib_from_scratch: A work-in-progress to generate the intermittent encoders from the IB-optimization process. Meant for 
future development, current results are suspect. 

res: houses the encoders, quasi-convexity values and other results used to generate the charts and statistics cited in 
the paper. 

The IB color-naming modeling framework is taken from 

@article{Zaslavsky2018efficient,
    author = {Zaslavsky, Noga and Kemp, Charles and Regier, Terry and Tishby, Naftali},
    title = {Efficient compression in color naming and its evolution},
    journal = {Proceedings of the National Academy of Sciences}
    volume = {115},
    number = {31},
    pages = {7937--7942},
    year = {2018},
    doi = {10.1073/pnas.1800521115},
    publisher = {National Academy of Sciences},
    issn = {0027-8424}
}

This repository will eventually house the final paper.

## How to run the code

''
