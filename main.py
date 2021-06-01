import os
from argparse import ArgumentParser
from datetime import datetime, timedelta
import numpy as np
import bdx_data as bd
import pandas as pd
import pytz
import time

import policy_iteration as poi
import expert_demo as ed


OUTPUT_LOC = '/app001/shared/'

parser = ArgumentParser(description='Main script for Eskind Setpoint Controller using Policy Iteration')
parser.add_argument('--output_loc', '-l', type=str, default=OUTPUT_LOC, help='Location to write setpoint for BdX')
parser.add_argument('--expert_demo', '-x', type=bool, default=True, help='Use expert heuristics to create some initial (few hours) data')

def xprt_dm(stpt_gtr, stpt_op_loc):
    if not stpt_op_loc.endswith('/'):
        stpt_op_loc += '/'
    # time at which to query the data
    query_time = datetime.now(tz=pytz.utc)  # have to provide current data in UTC time zone
    time_gap_minutes = 15
    start_time = query_time - timedelta(minutes=time_gap_minutes)
    try:
        df = bd.get_part_data(start_time,query_time,'4261')
        df.to_csv('ed_data/eskind_backup_v1_0.csv',index=False)
    except:
        df = pd.read_csv('ed_data/eskind_backup_v1_0.csv')
    oat,sah = df[['AHU_1 outdoorAirTemp','AHU_1 supplyAirHumidity']].iloc[-1,:].to_numpy()
    stpt = stpt_gtr.online_predictor(oat,sah=sah)

    with open(stpt_op_loc+'eskind_stpt_v1_0.csv', 'a') as cfile:
        cfile.write('{:.2f} \n'.format(stpt))
    cfile.close()

    return stpt_gtr

def pi_deployment(vi,stpt_gtr,last_VI,stpt_op_loc,xprt_demo):

    if xprt_demo:
        stpt_gtr = xprt_dm(stpt_gtr,stpt_op_loc)
    
    axn_candidates = np.linspace(55,62,100)
    
    # time at which to query the data
    query_time = datetime.now(tz=pytz.utc)  # have to provide current data in UTC time zone
    time_gap_minutes = 15
    start_time = query_time - timedelta(minutes=time_gap_minutes)
    try:
        df = bd.get_part_data(start_time,query_time,'4261')
        df.to_csv('vi_data/eskind_backup_v1_1.csv',index=False)
    except:
        df = pd.read_csv('vi_data/eskind_backup_v1_1.csv')
    df = poi.df_processor(df,to_save_path = 'vi_data/eskind_deployment_processed.csv')

    # get best setpoint using argmax(a)-->Q(s,a)
    df = vi.get_batch_data(path ='vi_data/eskind_deployment_processed.csv')
    s_a, _ = vi.create_s_a(df)
    s_a = s_a[-1,:].reshape(1,-1)
    s_a = s_a.repeat(100,axis=0)
    s_a[:,-1] = axn_candidates
    #get Q(s,a)
    Q_sa = vi.predict(s_a).flatten()
    # get best action 
    stpt = axn_candidates[np.argmax(Q_sa)]

    with open('vi_data/eskind_stpt_v1_1.csv', 'a') as cfile:
        cfile.write('{:.2f} \n'.format(stpt))
    cfile.close()

    if (datetime.now()-last_VI)>timedelta(hours=2):  # TODO: change it to 30 days
        # vi = ValIterFuncApprox(gamma=0.0,split=0.75)
        # train at the beginning
        poi.VI_train(vi)
        last_VI = datetime.now()  # (year=2021,month=5,day=1)
        checkpoint_path = "training_1/cp.ckpt"
        vi.load_weights(checkpoint_path)

    return [vi, stpt_gtr, last_VI]


if __name__ == "__main__":

    args = parser.parse_args()

    # Policy Iterator class pre-train
    os.makedirs('vi_data',exist_ok=True)
    vi = poi.ValIterFuncApprox(gamma=0.0,split=0.75)
    poi.VI_train(vi)
    last_VI = datetime.now()  # (year=2021,month=5,day=1)
    checkpoint_path = "training_1/cp.ckpt"
    vi.load_weights(checkpoint_path)
   

    os.makedirs('ed_data',exist_ok=True)
    stpt_gtr = ed.set_point_generator()

    while True:

        vi, stpt_gtr, last_VI=pi_deployment(vi, stpt_gtr, last_VI, stpt_op_loc=args.output_loc,xprt_demo=args.expert_demo)
        print("Completed a loop")
        time.sleep(timedelta(minutes=15).seconds)

        if datetime.now()>datetime(year=2021,month=6,day=1,hour=16):
            break
    print("Script Ended")

