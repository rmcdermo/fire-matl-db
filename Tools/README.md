### Usage

Required: Python 3.5

Currently, FMD includes a script for fitting TGA data using three methods.

To execute the script
```
$ python ./tga_fit.py (input file name) (heating rate in K/min)
```
For example,
```
$ python ./tga_fit.py UMD_PVC_3K_N2_1.csv 3
```
where a heating rate of 3 is input since the data was obtained at 3 K/min.

Output includes estimates of the kinetic parameters:
1. Number of reactions;
2. Pre-exponential factors (1/s);
3. Activation energies (kJ/mol); and
4. Mass changes for the reactions.
Additionally, plots of the residual mass fraction and the residual mass loss rates versus temperatures are generated.
