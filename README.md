# FormSA

This is a repository of python scripts to automate sensitivity analysis for thermoforming simulation using the commercial software [Aniform](https://aniform.com/)®.

To use the Python scripts provided in this repository, you will need a valid license for [Aniform](https://aniform.com/)® PrePost and Core.  

The following files are included in this repository: 

## runAniform.py
This is the main Python code that contain a few function definitions. 

To run this code, open a new terminal (for e.g., Windows PowerShell) and issue the following command inside the directory:
   
    python .\runAniform.py ''number of design samples'' ''Aniform input filename''

 A brief description of each of the functions included in *runAniform.py* are:

 (a) read_AniformModel(): This function uses the Pandas library to locate keywords within [Aniform](https://aniform.com/)® input file.
        
 (b) createDesignMatrix(): This function reads the ``MaterialmodeSens.txt'' file. The contains [Aniform](https://aniform.com/)® input file keywords for the Material models and associated parameters for which sensitivity analysis has to be performed. In other words, this is the file where user specify which are the design parameters, from Aniform material property definition, to be included in the sensitivity analysis. 

 The function then uses ``ChaosPy'' library to sample the parameter space for each of the specified parameter. To create the samples of a parameter, Uniform distribution is assumed. The mean of the distribution is taken as the value for the parameter specified in the [Aniform](https://aniform.com/)® input file. The sample space covered is calculated based on ten percentage deviation from the mean value. The samples for each parameter is then written to ''DesignPoints.xlsx'' file.
        
 (c) run_Aniform(): This function takes sample ID as argument. For each sample, a subdirectoy is created, and all required files for [Aniform](https://aniform.com/)® simulation is copied into the subdirectory. [Aniform](https://aniform.com/)® simulation is then run from terminal for that sample. 
