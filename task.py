# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:40:56 2020

@author: Krist
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing 
import time 
   

#%%
path = 'C:/Users/Krist/OneDrive/desktop/bbg/pj/'

book = pd.read_csv(path+'book_narrow_BTC-USD_2018.delim', sep = '|')
trade = pd.read_csv(path+'trades_narrow_BTC-USD_2018.delim', sep = '|')
#%%
book['spread'] = book['Ask1PriceMillionths'] - book['Bid1PriceMillionths']
book['Sratio'] = book['Bid1SizeBillionths'] / book['Ask1SizeBillionths']
#%%
book_seg1 = book.loc[(book['received_utc_nanoseconds'] >= 1522627200*1e9) & (book['received_utc_nanoseconds'] < (1522627200 + 5*24*3600)*1e9)].copy()
trade_seg1 = trade.loc[(trade['received_utc_nanoseconds'] >= 1522627200*1e9) & (trade['received_utc_nanoseconds'] < (1522627200 + 5*24*3600)*1e9)].copy()
#%%
def resample_price(x, n):
    if len(x) == 0:
        return np.nan
    else:
        l = book_seg1.loc[(book_seg1['received_utc_nanoseconds'] < \
                           book_seg1.loc[x[-1], 'received_utc_nanoseconds'] - 60*n*1e9),'Mid']
        return np.log(l.values[-1]) if len(l) > 0 else np.nan

def ewm_price(x):
    return book_seg1.loc[x,'Mid'].ewm(span = len(x)).mean().iloc[-1] if len(x) >= 1 else np.nan

def task(m):
    info = pd.DataFrame(index = trade_seg1.index, columns = [['spread_'+str(m)+'min','Sratio_'+str(m)+'min',\
                                                              'ewmp_'+str(m)+'min','vol_'+str(m)+'min']])
    spread = []
    Sratio = []
    ewmp = []
    vol = []
    for r in trade_seg1.index:
        print(r)
        t = trade_seg1.loc[r,'received_utc_nanoseconds']
        idx = book_seg1.loc[(book_seg1['received_utc_nanoseconds'] < t) \
              & (book_seg1['received_utc_nanoseconds'] >= t-60*m*1e9)].index
        spread.append(book_seg1.loc[idx,'spread'].mean())
        Sratio.append(book_seg1.loc[idx, 'Sratio'].mean())
        ewmp.append(ewm_price(idx))

        rsp = []
        for n in range(m*60//60+1):
            rsp.insert(0, resample_price(idx, n))
        var = 0
        for n in range(1,len(rsp)):
            var += (rsp[n] - rsp[n-1])**2
        v = var ** 0.5
        vol.append(v)
    info['spread_'+str(m)+'min'] = spread
    info['Sratio_'+str(m)+'min'] = Sratio
    info['ewmp_'+str(m)+'min'] = ewmp
    info['vol_'+str(m)+'min'] = vol
    info.to_csv(str(m)+'min.csv')
   
if __name__ == '__main__': 
    pool = multiprocessing.Pool(processes=4) 
    inputs = [2,5,10,30,60] 
    outputs = pool.map(task, inputs)