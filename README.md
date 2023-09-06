# BayeSFH-NGC0628
Python code used to infer the star formation history of "**The stellar population responsible for a kiloparsec-size superbubble seen in the JWST 'phantom' images of NGC 628**"

## Requirements

The present repo was developed using the following python packages:

- &nbsp; python 3.11.3
- &nbsp; Numpy 1.24.3
- &nbsp; Scipy 1.10.1
- &nbsp; pystan 2.7.0  ([Installation](https://pystan.readthedocs.io/en/latest/installation.html))

## Quick starting

If all the required packages are installed in your computer, execute the .py script by typing the following command line: python bayestar_3mags.py

If all packages work correctly, the program will run successfully and write an ASCII table with the following columns: Z, log_age, p10, p50, p90.

Plotting the second and fourth columns, you will get the following figure


<img src="https://github.com/Link4138/BayeSFH-NGC0628/blob/main/test.png" width="400"/>

## Citation
Please cite [Mayya, Y. D. et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.5492M/abstract) if this code was helpful for your research interest.
