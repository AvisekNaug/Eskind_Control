# Description
This repository creates an initial controller for deployment at Eskind Library.
The main set of commands to start the deployment is

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
```
conda activate eskind_pc
```

Upgrade protobuf(ignore pip's dependency resolver error as long as protobuf is successfully installed)
```bash
pip install --upgrade protobuf
```

Launch the script
```bash
nohup python main.py > main.out
```
