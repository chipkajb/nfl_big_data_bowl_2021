import os, sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()),'input/nfl-bdb-scripts'))
import pandas as pd
from pdb import set_trace as bp
from visualization import *
from training_data import *
from training_model import *
from testing_model import *
from analysis import *


#########################
###### ANIMATE PLAY #####
#########################
#animate_play(week=14, gameId=2018120911, playId=1584)


######################################
###### GET SPECIFIC PLAY METRICS #####
######################################
#get_specific_play_metrics(week=14, gameId=2018120911, playId=1584)


###################################
###### GENERATE TRAINING DATA #####
###################################
#generate_training_data()


########################
###### TRAIN MODEL #####
########################
#train_model()


####################################
###### TEST UNCALIBRATED MODEL #####
####################################
#model_path = "../input/nfl-bdb-data/best_0194_0.483.pt"
#test_uncalibrated_model(model_path)


############################
###### CALIBRATE MODEL #####
############################
#model_input_path = "../input/nfl-bdb-data/best_0194_0.483.pt"
#model_output_path = "../input/nfl-bdb-data/best_0194_0.483_calib.pt"
#calibrate_model(model_input_path, model_output_path)


##################################
###### TEST CALIBRATED MODEL #####
##################################
#model_path = "../input/nfl-bdb-data/best_0194_0.483_calib.pt"
#test_calibrated_model(model_path)


#######################################
###### CHECK CONTRIBUTION AMOUNTS #####
#######################################
#model_path = "../input/nfl-bdb-data/best_0194_0.483_calib.pt"
#check_contributions(model_path)


##########################################
###### GET ADVANCED COVERAGE METRICS #####
##########################################
#model_path = "../input/nfl-bdb-data/best_0194_0.483_calib.pt"
#get_advanced_metrics(model_path)


#############################################
##### ANALYZE ADVANCED COVERAGE METRICS #####
#############################################
pos = "lb" # "cb", "saf", or "lb"
analyze_advanced_metrics(pos)


###########################################
###### MAKE INTERACTIVE SCATTER PLOTS #####
###########################################
#scores_df = pd.read_csv('../input/nfl-bdb-data/cb_scores.csv')
#plot_playmaking_skills(scores_df)
#plot_coverage_skills(scores_df)
#plot_ball_skills(scores_df)
