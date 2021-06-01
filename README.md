# Description
This repository creates an initial controller for deployment at Eskind Library.
The main set of commands to start the deployment is

clone the repository
```bash
git clone git@github.com:AvisekNaug/Eskind_Control.git
```

navigate to the cloned repo
```bash
cd Eskind_Control
```

Create the conda environment
```bash
conda env create --file environment.yml
```

Launch the script
```bash
nohup python main.py --output_loc > main.out
```
