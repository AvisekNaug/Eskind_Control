import os
import pandas as pd
import numpy as np
np.random.seed(seed=123)
import bdx_data as bd
from datetime import datetime, timedelta
import pytz
import time

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
tf.random.set_seed(123)

class ValIterFuncApprox(tf.keras.Model):

    def __init__(self,output_shape=1,gamma=0.8,split=0.75):
        super(ValIterFuncApprox, self).__init__()
        
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        self.gamma = gamma
        self.split = split
        
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.dense4 = tf.keras.layers.Dense(output_shape, activation=tf.nn.relu)
        
        # compile model
        self.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        
        # callbacks
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        self.es_callback = tf.keras.callbacks.EarlyStopping(
                            # Stop training when `val_loss` is no longer improving
                            monitor="val_loss",
                            # "no longer improving" being defined as "no better than 1e-2 less"
                            min_delta=1e-3,
                            # "no longer improving" being further defined as "for at least 2 epochs"
                            patience=10,
                            verbose=1,
                            )

    def call(self, input_data):
        #input_data = tf.convert_to_tensor(input_data, dtype=tf.float64)
        x = self.dense1(input_data)
        x = self.norm1(x)
        x = self.dense2(x)
        x = self.norm2(x)
        x = self.dense3(x)
        x = self.norm3(x)
        x = self.dense4(x)
        return x

    def collect_rollout(self,batch_data_path='vi_data/eskind_pre_control_processed.csv'):
        # collect historical batch of BdX data
        df = self.get_batch_data(path=batch_data_path)
        
        # arrange them as a batches of|| states,actions(inputs),||
        s_a, s_prime = self.create_s_a(df) 
        
        # rewards(immediate reward)+current predictions for those states(Next state return)||
        expected_return = self.create_return(s_a,s_prime)
        
        # form input data and target data
        n = int(self.split*s_a.shape[0])
        self.X_train, self.X_val = s_a[:n,:],s_a[n:,:]
        self.Y_train, self.Y_val = expected_return[:n,:].flatten(),expected_return[n:,:].flatten()
        
    def get_batch_data(self,mode='offline',path = 'vi_data/eskind_pre_control_processed.csv'):
        if mode == 'offline':
            df =pd.read_csv(path)
            #df.drop(columns='time', inplace=True)
            df.dropna(inplace=True)
            # df = (df-df.max())/(df.max()-df.min()) 
        else:                                              # will implement online later
            pass
        
        return df
    
    def create_s_a(self,df):
        states = ['AHU_1 supplyAirTemp','AHU_1 supplyAirHumidity','AHU_1 mixedAirTemp',
                 'WeatherDataProfile humidity','total_kbtus']  # 
        # model will itself implement batch normalization
        actions = ['AHU_1 supplyAirTempSetpoint']
        
        s_a = df[states+actions].iloc[:-1,:].to_numpy()
        s_prime = df[states].iloc[1:,:].to_numpy()
        
        return [s_a,s_prime]
        
    def create_return(self,s_a,s_prime):
        
        # immediate reward
        r = immediate_reward(s_a).reshape(-1,1)
        state_value = []
        
        # return
        action_range = [55,62]
        action_candidates = np.linspace(action_range[0],action_range[1],100).reshape(-1,1)
        for i in s_prime:
            s = i.reshape(-1,i.shape[0]).repeat(100,axis=0)
            sa = np.concatenate((s,action_candidates),axis=1)
            state_value.append( np.mean(self.call(sa).numpy()) )
        
        state_value = np.array(state_value).reshape(-1,1)
        
        return r+self.gamma*state_value
    
    def iterate(self,):
        
        history = self.fit(
                self.X_train,self.Y_train,batch_size=64,shuffle=False,
                validation_data=(self.X_val, self.Y_val),
                epochs=100,callbacks = [self.cp_callback,self.es_callback]
                )
        
        return history

def immediate_reward(s_a):
    
    rh_set_point = 0.30*np.ones(shape=s_a.shape[0])
    rh_penalty = -1.0*np.abs(s_a[:,1]-rh_set_point)
    
    nrgy_penalty = -1.0*s_a[:,4]
    
    return rh_penalty+nrgy_penalty


def df_processor(df,to_save_path):
    df['total_fan_power'] = df['AHU_1 supplyFanVFDPower']+df['AHU_1 returnFanVFDPower'] \
                        +df['AHU_2 supplyFanVFDPower']+df['AHU_2 returnFanVFDPower'] \
                        +df['AHU_3 supplyFanVFDPower']+df['AHU_3 returnFanVFDPower']     
    j_2_btu = 0.000947817
    df['total_fan_kbtus'] = df['total_fan_power']*(5*60)*j_2_btu
    df['total_kbtus'] = df['total_fan_kbtus']+df['CHW_BTU_METER currentKbtuDeltaReading']+\
                        df['STM_BTU_METER currentKbtuDeltaReading']

    df.drop(columns=['AHU_1 returnFanVFDPower','AHU_1 supplyFanVFDPower',
    'AHU_2 returnFanVFDPower','AHU_2 supplyFanVFDPower','AHU_3 returnFanVFDPower','AHU_3 supplyFanVFDPower',
                    'total_fan_power','CHW_BTU_METER currentKbtuDeltaReading','total_fan_kbtus',
                    'STM_BTU_METER currentKbtuDeltaReading'],inplace=True)
    df.to_csv(to_save_path,index=False)

    return df

def VI_train(vi):
    # time at which to query the data
    query_time = datetime.now(tz=pytz.utc)  # have to provide current data in UTC time zone
    time_gap_days = 30
    start_time = query_time - timedelta(days=time_gap_days)
    try:
        df = bd.get_part_data(start_time,query_time,'4261')
        df.to_csv('vi_data/eskind_batch_vi_backup.csv',index=False)
    except:
        df = pd.read_csv('vi_data/eskind_batch_vi_backup.csv')
    df = df_processor(df, to_save_path = 'vi_data/eskind_batch_vi_processed.csv')
    df = vi.collect_rollout(batch_data_path='vi_data/eskind_batch_vi_processed.csv')
    vi.iterate()


if __name__ == "__main__":

    
    os.makedirs('vi_data',exist_ok=True)
    vi = ValIterFuncApprox(gamma=0.0,split=0.75)

    # train at the beginning
    VI_train(vi)
    last_VI = datetime.now()  # (year=2021,month=5,day=1)
    checkpoint_path = "training_1/cp.ckpt"
    vi.load_weights(checkpoint_path)

    axn_candidates = np.linspace(55,62,100)

    old_stpt = 0.0
    pred_once = False
    
    while True:

        # time at which to query the data
        query_time = datetime.now(tz=pytz.utc)  # have to provide current data in UTC time zone
        time_gap_minutes = 15
        start_time = query_time - timedelta(minutes=time_gap_minutes)
        try:
            df = bd.get_part_data(start_time,query_time,'4261')
            df.to_csv('vi_data/eskind_backup_v1_1.csv',index=False)
        except:
            df = pd.read_csv('vi_data/eskind_backup_v1_1.csv')
        df = df_processor(df,to_save_path = 'vi_data/eskind_deployment_processed.csv')

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
        if pred_once:
            stpt = old_stpt+np.clip(stpt-old_stpt,
                                     a_max=+0.09,a_min=-0.09) \
                                    +np.random.normal(0.0,0.2)
        old_stpt = stpt
        pred_once = True

        with open('../eskind_stpt_v1_1.csv', 'a') as cfile:
            cfile.write('{:.2f} \n'.format(stpt))
        cfile.close()

        if (datetime.now()-last_VI)>timedelta(days=1):
            # vi = ValIterFuncApprox(gamma=0.0,split=0.75)
            # train at the beginning
            VI_train(vi)
            last_VI = datetime.now()  # (year=2021,month=5,day=1)
            checkpoint_path = "training_1/cp.ckpt"
            vi.load_weights(checkpoint_path)
            continue
        time.sleep(timedelta(minutes=15).seconds)
        