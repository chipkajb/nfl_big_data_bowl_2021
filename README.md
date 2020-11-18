# NFL Big Data Bowl 2021

This repo contains football analytics scripts for the [NFL Big Data Bowl 2021](https://www.kaggle.com/c/nfl-big-data-bowl-2021).

Team members include: Jordan Chipka, Hunter Satterthwaite, Ryan Chipka, and Dan Chipka.

## Setup
Create the conda environment.
```
conda create -y -n nfl_bdb python=3.6
conda activate nfl_bdb
pip install -r requirements.txt
```

## Run
You can run a variety of Python scripts and Jupyter notebooks within this repo. Below is an example of how to run one of the scripts.

Activate the conda environment (if you have not already done so).
```
conda activate nfl_bdb
```

Enter the `scripts` directory and run the desired script.
```
cd scripts
python visualize_data.py
```
