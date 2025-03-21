# clmsThesis

## Overview

Repository for all things related to my M.S. thesis, 
\"Convexity is a Fundamental Feature of Efficient Semantic Compression in Probability Spaces.\"

This repository is designed to test the conditions under which IB optimized word mappings yield 
quasi-concave probability functions over the meaning space. 

In order to generate the natural language encoders you must first download the world color survey (WCS) data from 
https://linguistics.berkeley.edu/wcs/data.html. This data should be stored in a single directory that contains 
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

ib_from_scratch: A work-in-progress to generate the intermittent encoders from the IB-optimization process. Meant for 
future development, current results are suspect. This borrows heavily from https://github.com/nogazs/ib-color-naming and
https://gitlab.com/epiasini/embo.

res: houses the encoders, quasi-convexity values and other results used to generate the charts and statistics cited in 
the paper.

## How to run the code

1. Generate Natural Language Encoders:
   1. In order to perform the analyses discussed in \"Convexity is a Fundamental Feature of Efficient Semantic 
   Compression in Probability Spaces\" you must first download the WCS data and run wcs_data_processing.
   2. Then run natural_languages_assessment to generate the natural language encoders. 
      1. When you run natural_languages_assessment you must specify the appropriate directories at lines 21, 22, 26.
2. Generate IB Optimal Encoders:
   1. You must then download the optimal encoders and run get_optimal_encoders to generate the optimal encoders in the 
   correct format. 
      1. When you run get_optimal_encoders you must specify the appropriate directory at line 13.
3. Generate Sub-Optimal Encoders:
   1. Then you can run generate_suboptimal_encoders on both sets of encoders to generate the shuffled encoders. 
      1. When you run generate_suboptimal_encoders you will need to specify the appropriate filepaths at lines 216, 217, 
      218 and 224.
4. Generate Charts and Statistical Tests:
   1. Finally, run combined_chart_creation to generate the charts and statistical tests included in the paper. 
      1. When you run combined_chart_creation you will need to specify the output filepath at line 12.

''
