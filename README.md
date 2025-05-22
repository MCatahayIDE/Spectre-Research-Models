## Technical Methodology Report

### Conducting and Recording Spectre Attacks for Data Harvesting

#### Triggering A Spectre Attack
- **Adjust Default Attack Paramters:** Access and manipulate attack parameters from variables indicated in the Makefile.perf to change the default values of attack characteristics such as the attack interval, character length of attack, and sleep duration for idle performance logging.

- **Use FLAGS For Further Customization:** Choose between multiple modes of attack, each of which allow different parameters to be set in the terminal before initiating the attack. Currently there exists three attack modes along with the default (no parameters) mode for a total of four modes, selectable using the following syntax inside of "FLAGS = ..."
    - -ba <batch_size>        burst attack
    - -br <batch> <rate>      batch rate attack
    - -r  <rate>              rate attack
