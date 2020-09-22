#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:19:48 2020

@author: jingweizhang
"""

import requests
import pandas as pd
import time
from datetime import datetime

url_JUMP = "https://gbfs.uber.com/v1/dcs/free_bike_status.json"
url_staton_JUMP = "https://gbfs.uber.com/v1/dcs/station_status.json"
url_Lime = "https://data.lime.bike/api/partners/v1/gbfs/washington_dc/free_bike_status.json"
url_staton_Lime = "https://data.lime.bike/api/partners/v1/gbfs/washington_dc/station_status.json"
url_Lyft = "https://s3.amazonaws.com/lyft-lastmile-production-iad/lbs/dca/free_bike_status.json"
url_staton_Lyft = "https://s3.amazonaws.com/lyft-lastmile-production-iad/lbs/dca/station_status.json"
url_Spin = "https://web.spin.pm/api/gbfs/v1/washington_dc/free_bike_status"
url_bird = "https://gbfs.bird.co/dc"

#response = requests.get(url)
#df = pd.DataFrame(response.json()['data']['bikes'])
#df['Timestamp'] = response.json()['last_updated']
#
#response_staton = requests.get(url_staton)
#df_staton = pd.DataFrame(response_staton.json()['data']['stations'])
#
#df.to_csv('/data/locations' + str(response.json()['last_updated'])+'.csv')
#df_staton.to_csv('/data/stations' + str(response.json()['last_updated'])+'.csv')

def getdata(url, url_station, company, ttl):
    response = requests.get(url)
    Timestamp = datetime.now()
    
    if url_station:
        response_staton = requests.get(url_station)
    
    if count%ttl == 0:
        df = pd.DataFrame(response.json()['data']['bikes'])
        df['Timestamp'] = Timestamp
        df.to_csv('./data/' + company + '/locations/' + Timestamp.strftime("%Y:%m:%d-%H:%M:%S") +'.csv')
        if url_station:
            df_staton = pd.DataFrame(response_staton.json()['data']['stations'])
            df_staton.to_csv('./data/' + company + '/stations/' + Timestamp.strftime("%Y:%m:%d-%H:%M:%S") +'.csv')

starttime=time.time()

count = 0
while count < 129600:
    print(datetime.now())
    time.sleep(30.0 - ((time.time() - starttime) % 30.0))
    try:
        getdata(url_JUMP, url_staton_JUMP, "JUMP", 2)
    except:
        pass
    try:
        getdata(url_Lime, url_staton_Lime, "Lime", 2)
    except:
        pass
    try:
        getdata(url_Lyft, url_staton_Lyft, "Lyft", 2)   
    except:
        pass 
    try:
        getdata(url_Spin, False, "Spin", 2)    
    except:
        pass
    try:
        getdata(url_bird, False, "Bird", 2)    
    except:
        pass
    
    count = count + 1
        
    
    
    
#    response = requests.get(url)
#    df = pd.DataFrame(response.json()['data']['bikes'])
#    df['Timestamp'] = response.json()['last_updated']
#
#    df_staton = requests.get(url_staton)
#    df_staton = pd.DataFrame(response_staton.json()['data']['stations'])
#    
#    df.to_csv(str(response.json()['last_updated'])+'location.csv')
#    df_staton.to_csv(str(response.json()['last_updated'])+'location_staton.csv')
#    count = count + 1
    