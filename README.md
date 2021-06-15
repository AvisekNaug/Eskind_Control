# Description
This repository creates an initial controller for deployment at Eskind Library.
The main set of commands to start the deployment is

**Please note that either conda or miniconda being used has 64 bit arch.**

clone the repository
```bash
git clone https://github.com/AvisekNaug/Eskind_Control.git
```

navigate to the cloned repo
```bash
cd Eskind_Control
```

Create the conda environment
```bash
conda env create --file environment.yml
```

Activate the environment
```bash
conda activate eskind_pc
```

Launch the script
```bash
nohup python main.py > main.out
```

In the above command you can optionally use --output_loc argument to specify where to output the csv file.
Default location is '/app001/shared/' similar to Alumni Hall deployment