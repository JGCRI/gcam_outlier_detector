# GCAM Outlier Detector

This program finds data outliers in a GCAM database by comparing historical and generated data. By default, data until 2020 is assumed to be historical data. This end of historical year can be updated if needed. An aggregate stat like mean/median/quartiles is calculated from the historical data and it's checked whether any data point in the generated data is more than a certain threshold (by default 2x) of the aggregated stat. The data point in the generated data is marked as an outlier if this is the case. 

## On HPC/Deception

Since the GCAM database is large and can take time to be read, the outlier detector runs well on an HPC system, especially if multiple queries need to be processed. To run the detector on HPC, apptainer is needed which is present as a module load command in most HPCs. 

After cloning this repo on the HPC system, the following commands can be used to first build the container needed then run the detector inside the container.

```bash
[user@hpc gcam_outlier_detector]$ apptainer build --force --fakeroot container.sif container.def
[user@hpc gcam_outlier_detector]$ apptainer run --writable-tmpfs --fakeroot --userns --compat --bind <path-to-database-directory>:/databases --bind <path-for-storing-results>:/data --cwd /gcam_outlier_detector container.sif -d /databases/<database-name> -n -1 --csv-path /data --graph-path /data <any-other-args>
```

The command above will run the detector on all included queries with default labels. Notice how the csv path and graph path is explicitly provided. Apptainer does not retain any data once the container dies so to save outputs from the detector, a **binded** directory's path should be provided. If the entire *gcam_outlier_detector* directory is binded then passing explicit paths may not be needed since the default directory to save data is the current working directory. 

## Local Computer

While it can take a while to run on a personal/local computer, it is still possible and made quite easy with Docker. The *docker-compose.yml* in the repo can be modified to change the arguments to the detector or the path of the database attached. The `command` and `volumes` section of the compose file needs to be modified according to your needs and local directory names. After doing so, running the following command should be enough.

```bash
[user@localhost gcam_outlier_detector]$ docker compose up --build
```