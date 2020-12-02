# NFL Big Data Bowl 2021
## AI-Based Evaluation of NFL Pass Defenders

This repo was developed for the 2021 NFL Big Data Bowl. You can view the resulting report [here](https://www.kaggle.com/chipkajb/nfl-big-data-bowl-2021).

### Setup
You will first need to download the Big Data Bowl [data](https://www.kaggle.com/c/nfl-big-data-bowl-2021/data) to the
`input/nfl-big-data-bowl-2021` directory.

Make sure that all of the requirements are met.
```
pip install -r requirements.txt
```

If you are not working on a computer with a GPU, then the above command might fail when attempting to install `torch`.
If this is the case, you will need to comment out `torch` in `requirements.txt`, and then run the following commands to create the necessary conda environment.
```
conda create -n nfl_bdb python=3.6
conda activate nfl_bdb
conda install pytorch-cpu -c pytorch
pip install -r requirements.txt
```

### Run
You can run everything from `main.py` located in the `main` directory. You can simply comment/uncomment the sections that you
would like to run.
