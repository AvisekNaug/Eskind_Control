from datetime import datetime, timedelta, timezone
import pytz
from bdx import get_trend
import numpy as np

def pull_online_data(*args, **kwargs):
    """
    This method pulls data from the Niagara Networks using the BdX API
    """

    tz_utc = pytz.timezone("UTC")#timezone(offset=-timedelta(hours=0))
    #tz_central = timezone(offset=-timedelta(hours=6))

    start = datetime(kwargs['start_year'], kwargs['start_month'], kwargs['start_day'], hour= kwargs['start_hour'],
     minute= kwargs['start_minute'], second= kwargs['start_second'], tzinfo=tz_utc)#=====modifications
    end   = datetime(kwargs['end_year'], kwargs['end_month'], kwargs['end_day'], hour= kwargs['end_hour'],
     minute= kwargs['end_minute'], second= kwargs['end_second'], tzinfo=tz_utc)#=====modifications

    dataframe = get_trend(trend_id=kwargs['trend_id'],
                          username=kwargs['username'],
                          password=kwargs['password'],
                          start=start,
                          end=end)

    return dataframe

def get_part_data(start,end,trend_id):
    
    start_fields = ['start_'+i for i in ['year','month','day', 'hour', 'minute', 'second']]
    end_fields = ['end_'+i for i in ['year','month','day', 'hour', 'minute', 'second']]
    time_args = {}
    for idx, i in enumerate(start_fields):
        time_args[i] = start.timetuple()[idx]
    for idx, i in enumerate(end_fields):
        time_args[i] = end.timetuple()[idx]    
    api_args = {"trend_id" : trend_id, "username": "naugav","password": "NaugP@$$"}
    api_args.update(time_args)
    # get the data
    return pull_online_data(**api_args) 