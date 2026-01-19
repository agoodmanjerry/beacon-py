# Parameter Estimation(PE) on the cleaned data by seqARIMA.

0. Login to CIT clurster.

1. Activate the conda environment.  
    Go to environments directory and activate it.
    ```
    cd /cvmfs/software.igwn.org/conda/envs
    conda activate ./igwn-py310-20240729
    ```

2. Authetication  
    Run the following commands:
    ```
    kinit
    export HTGETTOKENOPTS="-a vault.ligo.org -i igwn"
    htgettoken
    ```
    For any issues, refer to: https://computing.docs.ligo.org/guide/auth/

3. Make the following chnages to the `.ini file`.
    ```
    data-dict= {data file path of your choice for each detector}
	psd-dict= {data file path of your choice for each detector}
	accounting-user= {your_username, e.g. albert.einstein}
	label= {label_name, e.g. seqARIMA} (Just for convenience in identifying tasks)
	outdir= {outdir directory under your home, e.g. /home/albert.einstein/pe/04/S230522n/outdir}
	webdir= {webdir directory under your home, e.g. /home/albert.einstein/public_html/pe/04/S230522n}
    ```
    (The `.ini file` will automatically create the outdir and webdir directories.)  
    (The PE results will be available at: https://ldas-jobs.ligo.caltech.edu/~your_username/)<br><br>
      

    You also need to comment out the following arguments if they are not None:	
    ```
    distance-marginalization-lookup-table = {DIR}
	waveform-arguments-dict
    ```

    <br>Since this is a test job, change queue to None,
    ```
    queue= None
    ```
    or just comment out like this:
    ```
    # queue=Online_PE
    ```

4. Submit the .ini file after making the above changes to it.
    ```
    bilby_pipe your_config.ini -- submit
    ```