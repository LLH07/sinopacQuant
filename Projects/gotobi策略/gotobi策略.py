## current version:
## test versiom:
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *
from scipy.stats import gmean
from xbbg import blp
import sys
# sys.path.append(r'\\10.5.20.31\交易部\Blair\backTestTool')
# sys.path.append(r'\\10.5.20.31\交易部\FX trading Strategy souce code\Wilson\程式碼')

import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False

import sys
from contextlib import contextmanager
import io
import os

# from backtestModuleSteveVer import (DrawDown, calculate_detailed_mdd, generate_recent_returns_df, 
#                                     print_stats, visualize_portfolio, format_position, 
#                                     calculate_portfolio_value,process_portfolio_data)
# import backtestModuleVer2_1128
import backtestModuleVer_20251002

# Add this at the beginning of your code, after the imports
strgyName = '1105'
traderName = 'Blair'

devCcy =  ['EUR', 'JPY', 'GBP', 'CHF', 'AUD', 'CAD', 'NZD', 'SEK']
revL = ['NZD', 'EUR', 'AUD', 'GBP']
basket = devCcy
#########################################
from datetime import datetime
today = datetime.now()+pd.Timedelta(days=0)
today_add5 = datetime.now()+pd.Timedelta(days=5)

today_str = today.strftime('%Y-%m-%d')
today_add5_str = today_add5.strftime('%Y-%m-%d')
# startDate = today-pd.Timedelta(days=600)
startDate = '2012-01-01'
endDate = today_add5_str

dr = pd.date_range(
    start=pd.to_datetime(startDate),
    end=pd.to_datetime(endDate),
    freq=BDay()
)  

gotobiL = [5, 10, 15, 20, 25, 30]
gotobiD = dict.fromkeys(gotobiL)

'''
根據https://oec.world/en/profile/country/jpn?yearlyTradeFlowSelector=flow1的資料，日本自: 中國、美國、德國以及澳洲進口商品
'''
jpImportCcys = ['USDJPY', 'CNHJPY', 'EURJPY', 'AUDJPY']
basket = jpImportCcys

import pandas as pd
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar, Holiday
import numpy as np

def adjust_gotobi_dates(dr, gotobiL, openOrClose, calendar=None):
    """
    Adjusts Gotobi dates to account for weekends and holidays.
    
    Parameters:
    dr (DatetimeIndex): Date range for the strategy
    gotobiL (list): List of Gotobi days (5, 10, 15, 20, 25, 30)
    calendar (CustomBusinessDay, optional): Custom business day calendar with holidays
    
    Returns:
    tuple: (open_dates, close_dates) - dictionaries mapping original dates to adjusted dates
    """
    # Initialize calendars
    if calendar is None:
        # Default to US Federal holidays if no calendar provided
        calendar = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
    
    # Create business day offset for calculations
    bday = BDay()
    
    # Extract all dates from dr
    all_dates = dr.to_list()
    
    # Create dictionaries to store original and adjusted dates
    positions = {}  # For opening positions (day before Gotobi)
    
    if openOrClose == 'open':
    # Process each month in the date range
        for date in dr:
            # Check if the day is a Gotobi target day
            day = date.day
            if day in gotobiL:
                # Original position opening date (day before Gotobi)
                day_before = date - pd.Timedelta(days=1)
                
                # Adjust if it falls on weekend or holiday
                while day_before not in all_dates or day_before.dayofweek >= 5:
                    day_before = day_before - pd.Timedelta(days=1)
                
                # Apply business day adjustment to ensure it's a valid trading day
                day_before = day_before.normalize()
                day_before_adjusted = day_before
                while day_before_adjusted not in all_dates:
                    day_before_adjusted = (day_before_adjusted - bday).normalize()
                positions[date] = day_before_adjusted

    max_steps = 10  # set a reasonable cap
    if openOrClose == 'close':
        for date in dr:
            day = date.day
            if day in gotobiL:
                day_of = date
                steps = 0

                # Adjust for weekends and holidays
                while (day_of not in all_dates or day_of.dayofweek >= 5) and steps < max_steps:
                    day_of = day_of + pd.Timedelta(days=1)
                    steps += 1
                if steps == max_steps:
                    print(f"Warning: Could not find a valid trading day for {date}")
                    continue

                # Apply business day adjustment to ensure it's a valid trading day
                day_of = day_of.normalize()
                day_of_adjusted = day_of
                steps = 0
                while day_of_adjusted not in all_dates and steps < max_steps:
                    day_of_adjusted = (day_of_adjusted + bday).normalize()
                    steps += 1
                if steps == max_steps:
                    print(f"Warning: Could not adjust to a valid trading day for {date}")
                    continue

                positions[date] = day_of_adjusted
    return positions



from scipy.stats import skew, kurtosis

gotobiDayOpenD = dict.fromkeys(gotobiL)
gotobiDayCloseD = dict.fromkeys(gotobiL)
tmpDr = pd.date_range(
    start=pd.to_datetime(startDate),
    end=pd.to_datetime(endDate),
    freq=Day()
)  

for d in gotobiL:
    tmp = [d]
    gotobiDayOpenD[d] = adjust_gotobi_dates(dr=tmpDr, openOrClose='open', gotobiL=tmp).values()
    gotobiDayCloseD[d] = adjust_gotobi_dates(dr=tmpDr, openOrClose='close', gotobiL=tmp).values()



import my
spot = my.getCMTKSpot(basket, startDate, endDate)

### new signal: add cross JPY
signalDf = pd.DataFrame(0, index=dr, columns=basket)

gotobiDay_all_open = list(gotobiDayOpenD[5]) + list(gotobiDayOpenD[10]) + list(gotobiDayOpenD[15]) + list(gotobiDayOpenD[20]) + list(gotobiDayOpenD[25]) + list(gotobiDayOpenD[30]) 
gotobiDay_all_close = list(gotobiDayCloseD[5]) + list(gotobiDayCloseD[10]) + list(gotobiDayCloseD[15]) + list(gotobiDayCloseD[20]) + list(gotobiDayCloseD[25]) + list(gotobiDayCloseD[30]) 

signalDf.loc[signalDf.index.isin(gotobiDay_all_open), :] = 1/len(basket)
signalDf.loc[signalDf.index.isin(gotobiDay_all_close), :] = 0



dxy = blp.bdh(tickers='DXY Index', flds = ["PX_LAST"], start_date=startDate, end_date=endDate).iloc[:]                                  
dxy.index = pd.DatetimeIndex(dxy.index)
dxy = dxy.reindex(dr).ffill()
dxyMa = dxy.rolling(20).mean()

# # 在DXY與USDJPY走勢高度相關且DXY在走弱的時候不進場
corrWindow = 20
corr = dxy.fillna(method='ffill').iloc[:, 0].rolling(corrWindow).corr(spot['USDJPY'].fillna(method='ffill'))


corrFilter = corr[corr > 0.7].dropna().index
dxyFilter = dxy[(dxy < dxyMa)].dropna().index

totalFilter = corrFilter.intersection(dxyFilter)

signalDf2 = signalDf.copy()
signalDf_withFilter = signalDf.copy()  
signalDf_withFilter.loc[signalDf_withFilter.index.isin(totalFilter), :] = 0

if (today_str in signalDf_withFilter.index) and (signalDf2.loc[today_str, 'USDJPY'] != 0):
    print(f'今日原有gotobi信號，但訊號濾除: DXY: {dxy.iloc[-1][0]}; DXY移動平均: {dxyMa.iloc[-1][0]}; DXY與USDJPY相關係數: {round(corr.iloc[-1], 2)} (高相關且DXY走弱，不開倉)')

# convert cross JPY signal to major currency
signalNormal= pd.DataFrame(0, index=signalDf_withFilter.index, columns=['JPY', 'CNH', 'AUD', 'EUR'])
for cross in signalDf_withFilter.columns:
    if cross == 'USDJPY':
        signalNormal['JPY'] = signalNormal['JPY'] + signalDf_withFilter['USDJPY']
    elif cross == 'CNHJPY':
        firstCcy = cross[:3]
        signalNormal[firstCcy] = signalNormal[firstCcy] - signalDf_withFilter[cross]
        signalNormal['JPY'] = signalNormal['JPY'] + signalDf_withFilter[cross]
    else:
        firstCcy = cross[:3]
        signalNormal[firstCcy] = signalNormal[firstCcy] + signalDf_withFilter[cross]
        signalNormal['JPY'] = signalNormal['JPY'] + signalDf_withFilter[cross]


# signalNormal.to_csv('gotobiTmp.csv')
# signalDf_final = pd.read_csv('gotobiTmp.csv')

signalNormal = signalNormal[signalNormal.index <= today_str]

signalDf = signalNormal.copy()
weight_size = 4.7
weight_size_tmp = 4.7/ 1.75
signalDf = signalDf * weight_size_tmp
signalDf.to_csv(f'may2025Ver_gotobiCrossSignal.scv')

#########################################################################################################################################
# 準備區域
def print_action_signals(action_row, ticker_mapping):
    exchange_rate_pairs = {}
    signals = []
    
    # 自動獲取匯率
    for ticker, action in action_row.items():
        if pd.notna(action) and action != 0 and ticker in ['EUR', 'GBP', 'AUD', 'NZD']:
            # 構建 Bloomberg ticker
            bloomberg_ticker = f"{ticker_mapping.get(ticker, ticker)} CMTK Curncy"
            # 獲取最新匯率
            try:
                rate_data = blp.bdh(tickers=bloomberg_ticker, flds=['PX_LAST'], start_date=datetime.now().date())
                exchange_rate = float(rate_data.xs('PX_LAST', axis=1, level=1).iloc[-1].values[0])
                exchange_rate_pairs[ticker] = exchange_rate
            except Exception as e:
                print(f"無法獲取 {ticker} 匯率: {str(e)}")
                exchange_rate_pairs[ticker] = 1.0  # 設置默認值
    
    # 處理交易信號
    for ticker, action in action_row.items():
        if pd.notna(action) and action != 0:
            action_str = "Long" if action > 0 else "Short"
            quote = ticker_mapping.get(ticker, ticker)
            if ticker in exchange_rate_pairs:
                adjusted_amount = action / exchange_rate_pairs[ticker]
            else:
                adjusted_amount = action
            mio_suffix = "mio" if ticker not in exchange_rate_pairs else f"mio in {ticker} ({exchange_rate_pairs[ticker]:.4f})"
            signal_text = f"{action_str} {quote} {abs(adjusted_amount):.3f} {mio_suffix}"
            print(signal_text)
            signals.append(signal_text)
            
    return signals

# 貨幣對映射
ticker_mapping = {
    'EUR': 'EURUSD', 'AUD': 'AUDUSD', 'JPY': 'USDJPY','CNH': 'USDCNH'}

#########################################################################################################################################
# 處理信號
today_date = signalDf.index.max()
yesterday_date = signalDf[signalDf.index < today_date].index.max()

df_yesterday = signalDf.shift(1).loc[today_date].to_frame().T
df_today = signalDf.loc[today_date].to_frame().T
df_action = (signalDf.loc[today_date] - signalDf.shift(1).loc[today_date]).to_frame().T

# 設置索引
df_yesterday.index = [yesterday_date]
df_today.index = [today_date]
df_action.index = ['Action']
# 合併最終表格
df_final = pd.concat([df_yesterday, df_today, df_action])

# 格式化日期索引
df_final.index = df_final.index.map(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x)

# 顯示結果
print("\n持倉狀況：")
pd.options.display.float_format = "{:.3f}".format
print(df_final)
print("\nAll in Pair Direction.")
print(f'以上為Cross Gotobi策略\n')
action_row = df_final.iloc[-1]
signals = print_action_signals(action_row, ticker_mapping)

import sys
# sys.path.append(r'\\10.5.20.31\交易部\Blair\backTestTool')
# sys.path.append(r'\\10.5.20.31\交易部\FX trading Strategy souce code\Wilson\程式碼')
# import my_function as my_Wilson

startDate = "2025-01-02"
signalDf['Date'] = signalDf.index
last_date = signalDf["Date"].iloc[-1]
# bt = backtestModuleVer2_1128.backTest(MIO=weight_size, signalDf=signalDf, startDate=startDate, endDate=endDate)
bt = backtestModuleVer_20251002.backTest(MIO=weight_size, signalDf=signalDf, startDate=startDate, endDate=endDate)
bt.backtestFun()
bt.strgyStats()

tmpStart = '2025-01-01'
# bt.portValChange[bt.portValChange.index >= tmpStart].to_csv(f'2025_dailyRet_1105.csv')

n = 10
Sharpe_ratio = round(bt.sharpe, 2)
value_profit = round(bt.portValCumChange.iloc[-1], 2)
value_percentage = round((bt.portCumChange.iloc[-1] - 1)*100, 2)
Mdd_current = round((bt.mdd.min()*100), 2)
temp_df = bt.portChange.copy()

Current_pct = round((temp_df*100).to_frame(), 3)
Current_pct = Current_pct.rename(columns={0: f'近{n}日報酬率變化'})
Current_pct[f"近{n}日報酬率變化(USD)"] = Current_pct[f'近{n}日報酬率變化']/100*weight_size*1_000_000
Current_pct['本月累積損益(USD)'] = Current_pct.groupby(pd.Grouper(freq='M'))[f'近{n}日報酬率變化(USD)'].cumsum()

startDate = "2025-01-01"
print(f"回測時間起始: {bt.startDate}")
print(f"Sharpe_Ratio : {Sharpe_ratio}")
print(f"總報酬率 : {value_percentage}%")
print(f"總報酬金額 : {value_profit} USD")
print(f"今年最大回撤 : {Mdd_current}%")
print("\n")
print(Current_pct.tail(n))

# 儲存數據
use_df = pd.DataFrame()
use_df["Date"] = bt.portChange.index
use_df["Portfolio_Return"] = bt.portChange.values
use_df["Portfolio_Return_Profit"] = use_df["Portfolio_Return"]*weight_size*1_000_000
use_df["Year"] = use_df["Date"].dt.year
use_df["Month"] = use_df["Date"].dt.month

daily_return = use_df[["Portfolio_Return"]].copy()
daily_return["Cumsum"] = daily_return["Portfolio_Return"].cumsum()
daily_return["Date"] = use_df["Date"]

# 整理回測結果
backtest_results = {
    'total_return': (bt.portCumChange.iloc[-1] - 1)*100,
    'total_profit': bt.portValCumChange.iloc[-1],
    'max_drawdown': round((bt.mdd.min()*100), 2),
    'sharpe_ratio': round(bt.sharpe, 2)
}

strategy_name = "Gotobi"
murex_code = 1105

# # 保存HTML
# today = datetime.now().strftime('%Y%m%d')
# save_path = f'\\\\10.5.20.31\交易部\\FX trading Strategy souce code\\Quant_Portfolio\\Original_Data_Renew\\HTML_Report\\11_Blair\\{strategy_name}_{today}.html'

# my_Wilson.save_strategy_html(df_final, signals, backtest_results, Current_pct, save_path, murex_code=murex_code, strategy_name=strategy_name)

# # 先處理存儲資料的部分
# daily_total = round((bt.portChange).to_frame(name="Portfolio_Return_Profit"),5)
# daily_total["Portfolio_Return_Profit"] = daily_total["Portfolio_Return_Profit"]*weight_size*1_000_000
# daily_total = daily_total.loc[:today]
# my_Wilson.export_strategy_position(signalDf,strategy_name)
# my_Wilson.update_excel_save_excel(1105, daily_total)

