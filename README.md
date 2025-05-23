## Technical Methodology Report

### Conducting and Recording Spectre Attacks for Data Harvesting

### Triggering A Spectre Attack

- **Adjust Default Attack Paramters:** Access and manipulate attack parameters from variables indicated in the Makefile.perf to change the default values of attack characteristics such as the attack interval, character length of attack, and sleep duration for idle performance logging.

- **Use FLAGS For Further Customization:** Choose between multiple modes of attack, each of which allow different parameters to be set in the terminal before initiating the attack. Currently there exists three attack modes along with the default (no parameters) mode for a total of four modes, selectable using the following syntax inside of "FLAGS = ..."
    - -ba <batch_size>        burst attack
    - -br <batch> <rate>      batch rate attack
    - -r  <rate>              rate attack

- **Ensure Proper Formatting In CSV Master Track File** Once an attack batch is complete, a directory path is generated for its .txt file and then stored in a master track file to keep track of any txt file data that needs to be converted to CSV. However, upon generating new directory paths, the underlying function inserts them with a single-space indent, making them harder to detect by the CSV converter function. Thus once new directory paths are generated within the master track file (CSV), simply any entries generated with spaces before them to prevent any issues with CSV conversion. Further, if a new batch of data is generated but no new directories generate in the master track file, simply force a save to trigger a version comparison that will prompt you with conflicting versions of the master track file. From there, revert to the version that reflects the new directory paths from your latest trial.

### Converting .txt data files to .csv

- **Use Sampler_final.py convertCSV Function** The CSV file conversion uses the master track CSV to track which .txt files need to be converted into CSVs, so the function simply needs to be executed in the terminal. It will also be able to ignore any files it's detected have already been detected. New CSV files should appear within the coressponding CSV file for the coressponding timestamp trial batch group.

### Unify A Timestamp's batch CSV files Into A Single CSV
- **Use Jupyter Combine_Batch_File Functions** Iterate through the first jupyter file modules to import relevant packages. Then locate the combine_csv function. Near the bottom of the function locate the directory path input and change the directory to locate the folder containing all CSVs for a given timestamp for conversion. Similarly, change the output directory name and path to direct the output to CSV to where you need it

