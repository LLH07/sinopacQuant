class backTest():
    def __init__(self, MIO, signalDf, startDate, endDate=None, strgyCode=None):
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        from datetime import date,datetime, timedelta
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        

        if endDate == None:
            today = datetime.now()+pd.Timedelta(days=0)
            today = today.strftime('%Y-%m-%d')
            self.endDate = today
        else: 
            self.endDate = endDate

        self.strgyCode = strgyCode
        self.MIO = MIO
        self.startDate = startDate

        test = signalDf
        test.rename(columns={'Unnamed: 0':'Date'}, inplace=True)
        test.index = test['Date']
        test.drop(columns='Date', inplace=True)
        test.index = pd.DatetimeIndex(test.index)
        self.signal = test

        self.dailyPrice = self.calSpotSwapRate()[0]
        self.dailySwap = self.calSpotSwapRate()[1]
        
        self.dailyPrice.loc['2025-04-01', 'INR'] = 85.5223
        self.dailyPrice.loc['2025-03-14', 'INR'] = 87.0653
        self.dailyPrice.loc['2025-02-26', 'INR'] = 87.1489
        self.dailyPrice.loc['2025-04-10', 'INR'] = 86.4730
        self.dailyPrice.loc['2025-04-18', 'INR'] = 85.3750

    def calSpotSwapRate(self):
        from pandas.tseries.offsets import BDay
        import pandas as pd
        from xbbg import blp

        # info = pd.read_excel(r'\\10.5.20.31\交易部\Blair\backTestTool\ccyInfo.xlsx')
        info = pd.read_excel(rf'C:\Users\Blair Lin\quantTrade_2025\utils\ccyInfo.xlsx')
        info = info[info['ccy'].isin(self.signal.columns)]
        info.index = info['ccy']

        rawDict = dict.fromkeys(['Spot', 'Swap'])
        dailyPrice = pd.DataFrame(columns=info.ccy)
        dailySwap = pd.DataFrame(columns=info.ccy)

        swapD = {
            'JPY':0.01,
            'KRW':1,
            'INR':0.01,
            'IDR':1,
            'TWD':0.001,
            'CZK':0.001,
            'THB':0.01,
            'HUF':0.01
        }

        for i in info.ccy:
            rawDict['Spot'] = blp.bdh(tickers=info.loc[i, 'bbgSpotTicker'], flds = ["PX_LAST"], start_date=self.startDate, end_date=self.endDate)
            rawDict['Swap'] = blp.bdh(tickers=info.loc[i, 'bbgSwapTicker'], flds = ["PX_LAST"], start_date=self.startDate, end_date=self.endDate)


            # calculate daily swap pts
            if i in swapD.keys():
                rawDict['Swap'] = rawDict['Swap']/30*swapD[i]
            else:
                rawDict['Swap'] = rawDict['Swap']/30*0.0001

            rawDict['Spot'].index = pd.DatetimeIndex(rawDict['Spot'].index)
            rawDict['Swap'].index = pd.DatetimeIndex(rawDict['Swap'].index)

            # reset the date range to period business day
            dr = pd.date_range(
                start=pd.to_datetime(self.startDate),
                end=pd.to_datetime(self.endDate),
                freq=BDay()
            )   

            rawDict['Spot'] = rawDict['Spot'].reindex(dr).ffill()
            rawDict['Swap'] = rawDict['Swap'].reindex(dr).ffill()

            dailyPrice[i] = rawDict['Spot']
            dailySwap[i] = rawDict['Swap']

        return dailyPrice, dailySwap
    
    def backtestFun(self):
        import pandas as pd
        import numpy as np
        spotDf = self.dailyPrice
        swapDf = self.dailySwap
        MIO = self.MIO
        conditionNotZero = self.signal.copy()
        conditionNotZero = conditionNotZero.loc[(conditionNotZero.index >= self.startDate) & (conditionNotZero.index <= self.endDate)]
        
        conditionNotZero.index = pd.DatetimeIndex(conditionNotZero.index)
        # check the period and ffill the signal
        conditionNotZero = conditionNotZero.resample('1B').ffill()
        conditionNotZero = conditionNotZero.shift(1)

        # conditionNotZero_diff = conditionNotZero - conditionNotZero.shift(1)
        conditionNotZero_diff = conditionNotZero.shift(-1) - conditionNotZero # 原來訊號-shift訊號
        self.signalChange = conditionNotZero_diff

        for j in conditionNotZero.columns:
            if j == 'EUR' or j == 'AUD' or j == 'NZD' or j == 'GBP':
                conditionNotZero.loc[:, j] = -1*conditionNotZero.loc[:, j]

        backtestSpot = pd.DataFrame(columns=conditionNotZero.columns, index=conditionNotZero.index)
        backtestSwap = pd.DataFrame(columns=conditionNotZero.columns, index=conditionNotZero.index)
        backtestSpotRaw = backtestSpot.copy()

        # 將所有匯率改為 USD 在左邊 
        for i in backtestSpot.columns:
            spotDf.loc[:, i] = spotDf.loc[:, i].astype('float')
            backtestSpot.loc[:, i] = spotDf.loc[:, i]
            backtestSpotRaw.loc[:, i] = spotDf.loc[:, i]
            backtestSwap.loc[:, i] = swapDf.loc[:, i]
            if i == 'EUR' or i == 'AUD' or i == 'NZD' or i == 'GBP':
                backtestSpot.loc[:, i] = 1/backtestSpot.loc[:, i]

        # 計算 swap points 損益
        swapSpotPercent =  backtestSwap / backtestSpotRaw  
        self.swapSpotPercent = swapSpotPercent
        
        # 計算 USD 損益
        backtestSpotShift = backtestSpot.shift(1)
        dailyChange = pd.DataFrame(index=backtestSpot.index, columns=backtestSpot.columns)
        dailyChange = (backtestSpot-backtestSpotShift)/backtestSpot
        self.spotMove = dailyChange
        self.portChange = pd.DataFrame(0, index=backtestSpot.index, columns=backtestSpot.columns)
        self.ccyPortChange = pd.DataFrame(0, index=backtestSpot.index, columns=backtestSpot.columns)
        
        
        slippage = 0.009 / 2
        for j in dailyChange.columns:
            # slippage = 0.0005 if j != 'JPY' else 0.05
            for i in dailyChange.index:
                
                if conditionNotZero.loc[i, j] > 0: # 做多
                    # 判斷贏錢還是輸錢
                    if dailyChange.loc[i, j] * conditionNotZero.loc[i, j] >= 0: # win
                        self.portChange.loc[i, j] = (dailyChange.loc[i, j] * (1-swapSpotPercent.loc[i, j]) * conditionNotZero.loc[i, j]) / MIO
                    else:
                        self.portChange.loc[i, j] = (dailyChange.loc[i, j] * (1-swapSpotPercent.loc[i, j]) * conditionNotZero.loc[i, j]) / MIO
                elif conditionNotZero.loc[i, j] < 0: # 做空
                    if dailyChange.loc[i, j] * conditionNotZero.loc[i, j] >= 0: # win
                        self.portChange.loc[i, j] = (dailyChange.loc[i, j] * (1+swapSpotPercent.loc[i, j]) * conditionNotZero.loc[i, j]) / MIO
                    else:
                        self.portChange.loc[i, j] = (dailyChange.loc[i, j] * (1+swapSpotPercent.loc[i, j]) * conditionNotZero.loc[i, j]) / MIO

                curIdx = conditionNotZero_diff.index.get_loc(i) #  and (conditionNotZero.loc[conditionNotZero.index[curIdx-1], j] != 0)
                if (conditionNotZero_diff.loc[i, j] != 0) and pd.notna(conditionNotZero_diff.loc[i, j]): # if new position opens or position closes # and (conditionNotZero.loc[conditionNotZero.index[curIdx-1], j] != 0)

                    self.portChange.loc[self.portChange.index[curIdx], j]  = self.portChange.loc[self.portChange.index[curIdx], j]  - slippage*0.01*abs(conditionNotZero_diff.loc[i, j])/MIO
                    # self.portChange.loc[self.portChange.index[curIdx-1], j]  = self.portChange.loc[self.portChange.index[curIdx-1], j]  - (slippage*abs(conditionNotZero_diff.loc[i, j]) / backtestSpot.loc[backtestSpot.index[curIdx-1], j])/MIO
                    # print(slippage, abs(conditionNotZero.loc[i, j]), backtestSpot.loc[backtestSpot.index[curIdx], j])
                    # print(self.portChange.loc[self.portChange.index[curIdx], j]  )
                    # print(f'after: {self.portChange.loc[self.portChange.index[curIdx], j]}')
                
            self.ccyPortChange[j] = self.portChange.loc[:, j]
            

        # slippage = 0.05 * 0.01  # Already 0.0005

        # for j in dailyChange.columns:
        #     for i in dailyChange.index:
                
        #         if conditionNotZero.loc[i, j] > 0: # Long position
        #             if dailyChange.loc[i, j] >= 0: # win
        #                 self.portChange.loc[i, j] = dailyChange.loc[i, j] * (1-swapSpotPercent.loc[i, j]) * conditionNotZero.loc[i, j]
        #             else: # lose
        #                 self.portChange.loc[i, j] = dailyChange.loc[i, j] * (1-swapSpotPercent.loc[i, j]) * conditionNotZero.loc[i, j]
                        
        #         elif conditionNotZero.loc[i, j] < 0: # Short position
        #             if dailyChange.loc[i, j] <= 0: # win (price goes down)
        #                 self.portChange.loc[i, j] = dailyChange.loc[i, j] * (1+swapSpotPercent.loc[i, j]) * conditionNotZero.loc[i, j]
        #             else: # lose
        #                 self.portChange.loc[i, j] = dailyChange.loc[i, j] * (1+swapSpotPercent.loc[i, j]) * conditionNotZero.loc[i, j]

        #         # Apply slippage when position changes
        #         if pd.notna(conditionNotZero_diff.loc[i, j]) and conditionNotZero_diff.loc[i, j] != 0:
        #             curIdx = conditionNotZero_diff.index.get_loc(i)
                    
        #             if curIdx > 0:  # Make sure we're not at the first index
        #                 yesterday_idx = self.portChange.index[curIdx-1]
        #                 yesterday_return = self.portChange.loc[yesterday_idx, j]
        #                 position_change_magnitude = abs(conditionNotZero_diff.loc[i, j])
                        
        #                 # Slippage cost proportional to position change
        #                 slippage_cost = position_change_magnitude * slippage
                        
        #                 # Apply slippage to yesterday's return (always a cost)
             
        #                 self.portChange.loc[yesterday_idx, j] = yesterday_return - slippage_cost

                        
        #                 print(f'{i} | {j} | Position change: {conditionNotZero_diff.loc[i, j]:.4f} | '
        #                     f'Yesterday return before: {yesterday_return:.6f} | '
        #                     f'Slippage: {slippage_cost:.6f} | '
        #                     f'After: {self.portChange.loc[yesterday_idx, j]:.6f}')
                
        #    self.ccyPortChange[j] = self.portChange.loc[:, j]

        # porValChange 計算回測損益 (絕對數字)、portChange 計算回測損益 (百分比)
        self.portValChange = self.portChange.copy()
        # self.portChange = (self.portChange/MIO).sum(axis=1)
        self.portChange = (self.portChange).sum(axis=1)
        # self.portValChange = ((self.portValChange/MIO).sum(axis=1))*MIO*1000000
        self.portValChange = ((self.portValChange).sum(axis=1))*MIO*1000000

        self.ccyPortValChange = self.ccyPortChange * MIO * 1_000_000
    
        self.portValCumChange = self.portValChange.cumsum() 
        self.portCumChange = self.portChange.cumsum() + 1
        self.condition = conditionNotZero


    
    def strgyStats(self):
        
        import portStats
        res = portStats.getStats(self.portChange.shape[0], self.portChange, self.portCumChange)
        self.mean = res[0]['Mean']
        self.std = res[0]['Std']
        self.sharpe = res[0]['Sharpe']
        self.mdd = res[1]

    def backTestPlot(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rc('font', family='Microsoft JhengHei')
        plt.rcParams['axes.unicode_minus'] = False

        if self.strgyCode != None:
            test3 = pd.read_excel('FOR0045彙整_Backup.xlsm', sheet_name='raw data')
            test3 = test3.loc[(test3['Date'] >= self.startDate) & (test3['Date'] <= self.endDate) & (test3['Portfolio'] == self.strgyCode)]
            test3 = test3.loc[:, ['Date', 'YTD PL USD']]
            test3.index = test3['Date']
            test3.index = pd.DatetimeIndex(test3.index)
            test3.drop(columns='Date', inplace=True)

            # 畫出策略的累積損益與 Murex 實際損益的比較圖
            commonIdx = list(self.portValCumChange.index.intersection(test3.index))
            resPlot = pd.Series(self.portValCumChange)

            resPlot = resPlot.loc[commonIdx]
            test3 = test3.loc[commonIdx]


            plt.figure(figsize=(15, 5))
            ax1 = plt.subplot(2, 1, 1)
            ax1.set_xlim([resPlot.index[0], resPlot.index[-1]])
            plt.plot(resPlot, label='回測損益', color='b')
            # ax2 = ax1.twinx()
            plt.plot(test3, label='Murex 實際損益', color='r')

            ax1.legend(loc='best')
            # ax2.legend(loc=1)
            plt.title(f'{self.strgyCode} 回測損益與實際損益')


            ax3 = plt.subplot(2, 1, 2)
            ax3.bar(self.mdd.index, self.mdd, width=10, label = 'MDD')
            ax3.set_xlim([self.mdd.index[0], self.mdd.index[-1]])
            ax3.legend(loc='best')
        else:
            resPlot = pd.Series(self.portValCumChange)
            plt.figure(figsize=(15, 5))
            ax1 = plt.subplot(2, 1, 1)
            ax1.set_xlim([resPlot.index[0], resPlot.index[-1]])
            plt.plot(resPlot, label='回測損益', color='b')
            ax1.legend(loc='best')
            plt.title(f'回測損益')

            ax3 = plt.subplot(2, 1, 2)
            ax3.bar(self.mdd.index, self.mdd, width=10, label = 'MDD')
            ax3.set_xlim([self.mdd.index[0], self.mdd.index[-1]])
            ax3.legend(loc='best')

