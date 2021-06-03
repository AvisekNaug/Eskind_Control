"""
This script generates some initial exploratory data for the set point
using a manually crafted exploratory controller so that the Value Iteration
can learn from some varied data. Otherwise, the VI would not learn anything
from a nearly constant data for the setpoint
"""
import os
import pandas as pd
import numpy as np
np.random.seed(seed=123)
import bdx_data as bd
from datetime import datetime, timedelta
import pytz
import time

# set point function generator for initial data
class set_point_generator():
    
    def __init__(self,):
        self.old_stpt = 0.0
        self.pred_once = False
    
    def online_predictor(self,oat,sah=50):
        if oat>= 65:
            stpt = 55.0
        elif (oat>=62.5) & (oat<65.0):
            stpt = 57.5
        elif (oat>=60.0) & (oat<62.5):
            stpt = 60.0
        elif (oat>=59.5) & (oat<60.0):
            stpt = 60.5
        else:
            stpt = 61.0
            
        if sah>60:
            stpt = stpt + 0.2

        if self.pred_once:
            stpt = self.old_stpt+np.clip(stpt-self.old_stpt,
                                         a_max=+0.30,a_min=-0.30) \
                                        +np.random.normal(0.0,0.2)
        self.old_stpt = stpt
        self.pred_once = True
        
        return stpt



if __name__ == "__main__":

    os.makedirs('ed_data',exist_ok=True)
    stpt_gtr = set_point_generator()

    while True:

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

        with open('../eskind_stpt_v1_0.csv', 'a') as cfile:
            cfile.write('{:.2f} \n'.format(stpt))
        cfile.close()

        time.sleep(timedelta(minutes=15).seconds)