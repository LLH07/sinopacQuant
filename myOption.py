import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import math
from scipy.stats import norm


class option():
    def __init__(self, ccy, tenor):
        self.ccyFull = ccy
        self.ccyLeft = ccy[:3]
        self.ccyRight = ccy[3:]
        if self.ccyLeft == 'USD':
            self.ccy = self.ccyRight
        elif self.ccyRight == 'USD':
            self.ccy = self.ccyLeft

        self.revL = ['EUR', 'GBP', 'AUD', 'NZD']

        rateCodeDict = {
            'USD': 'S0001M Index', # become sofr after 2024.09
            'EUR': 'EUR001M Index',
            'AUD': 'BBSW1M Index',
            'CNH': 'HICNH1M Index',
            'JPY': 'TI0001M Index'
        }

        dayCountDict = {
            'USD': 360, 
            'EUR': 360,
            'GBP': 365, 
            'JPY': 360, 
            'CHF': 360,
            'CAD': 360, 
            'NZD': 365,
            'AUD': 365,
            'CNH': 360,
            'ZAR': 365,
            'SGD': 365, 
            'SEK': 360
        }
        self.usdDayCount = dayCountDict['USD']
        self.otherCcyDayCount = dayCountDict[self.ccy]

        
        self.volCode = f'USD{self.ccy}V{tenor} Curncy' if self.ccy not in self.revL else f'{self.ccy}USDV{tenor} Curncy'
        self.histVolCode = f'USD{self.ccy}H{tenor} Curncy' if self.ccy not in self.revL else f'{self.ccy}USDH{tenor} Curncy'
        self.swapCode = f'USD{self.ccy}{tenor} Curncy' if self.ccy not in self.revL else f'{self.ccy}USD{tenor} Curncy'

        
    def getImpliedVol(self, startDate, endDate):
        from xbbg import blp
        dr = pd.date_range(
            start=pd.to_datetime(startDate),
            end=pd.to_datetime(endDate),
            freq=BDay()
        )
        return blp.bdh(self.volCode, 'PX_LAST', start_date=startDate, end_date=endDate, Per='D').iloc[:, 0].reindex(dr).ffill().shift(1).dropna()

    def getHistVol(self, startDate, endDate):
        from xbbg import blp
        dr = pd.date_range(
            start=pd.to_datetime(startDate),
            end=pd.to_datetime(endDate),
            freq=BDay()
        )
        return blp.bdh(self.histVolCode, 'PX_LAST', start_date=startDate, end_date=endDate, Per='D').iloc[:, 0].reindex(dr).ffill().shift(1).dropna()

    def getImpliedVolBasket(ccys, startDate, endDate):
        from xbbg import blp
        dr = pd.date_range(
            start=pd.to_datetime(startDate),
            end=pd.to_datetime(endDate),
            freq=BDay()
        )
        df = pd.DataFrame(index=dr, columns=ccys)
        for ccy in ccys:
            ccyVolCode = f'USD{ccy}V1M Curncy' if ccy not in ['EUR', 'GBP', 'AUD', 'NZD'] else f'{ccy}USDV1M Curncy'
            df[ccy] = blp.bdh(ccyVolCode, 'PX_LAST', start_date=startDate, end_date=endDate, Per='D').iloc[:, 0].reindex(dr).ffill().shift(1).dropna()

        return df
    

    
    # original function but have minor issues 
    def get30minBfixPrice(ccy, startDate, endDate):
        def shift_bfix_late_slots(df, late_threshold='16:30'):
            mapping_dict = {
                "T000": "23:00", "T003": "23:30", "T010": "00:00", "T013": "00:30", "T020": "01:00", "T023": "01:30",
                "T030": "02:00", "T033": "02:30", "T040": "03:00", "T043": "03:30", "T050": "04:00", "T053": "04:30",
                "T060": "05:00", "T063": "05:30", "T070": "06:00", "T073": "06:30", "T080": "07:00", "T083": "07:30",
                "T090": "08:00", "T093": "08:30", "T100": "09:00", "T103": "09:30", "T110": "10:00", "T113": "10:30",
                "T120": "11:00", "T123": "11:30", "T130": "12:00", "T133": "12:30", "T140": "13:00", "T143": "13:30",
                "T150": "14:00", "T153": "14:30", "T160": "15:00", "T163": "15:30", "T170": "16:00", "T173": "16:30",
                "T180": "17:00", "T183": "17:30", "T190": "18:00", "T193": "18:30", "T200": "19:00", "T203": "19:30",
                "T210": "20:00", "T213": "20:30", "T220": "21:00", "T223": "21:30", "T230": "22:00", "T233": "22:30"
            }

            df = df.copy()
            df = df.sort_index()
            df.index = pd.to_datetime(df.index)
            late_slots = [k for k, v in mapping_dict.items() if v >= late_threshold]
            late_times = [mapping_dict[slot] for slot in late_slots]
            all_dates = sorted({d.date() for d in df.index})
            for i, d in enumerate(all_dates[:-1]):
                today = pd.Timestamp(d)
                next_day = pd.Timestamp(all_dates[i+1])

                for t in late_times:
                    this_idx = pd.Timestamp(f"{today} {t}")
                    next_idx = pd.Timestamp(f"{next_day} {t}")
                    # get current slot 
                    currentSlot = next((k for k, v in mapping_dict.items() if v == t), None)
                    ccyStr = f'{ccy}USD' if ccy in ['EUR', 'GBP', 'AUD', 'NZD'] else f'USD{ccy}'
                    currentFull = f"{ccyStr} {currentSlot} Curncy"
                    # currentRate = blp.bdh(currentFull, "PX_LAST", this_idx.strftime('%Y-%m-%d'), next_idx.strftime('%Y-%m-%d'))
                    # prevRate = blp.bdh(currentFull, "PX_LAST", (this_idx - BDay(1)).strftime('%Y-%m-%d'), this_idx.strftime('%Y-%m-%d'))

                    # if currentRate.index[0].strftime('%Y-%m-%d') != this_idx.strftime('%Y-%m-%d'):
                    #     # print(f'Date not matched for : {currentRate.index[0].strftime("%Y-%m-%d")}, {this_idx.strftime("%Y-%m-%d")}')
                    #     df.loc[this_idx, "Last Price"] = np.nan
                    #     continue
                    if (next_idx in df.index) and (this_idx in df.index):
                        df.loc[this_idx, "Last Price"] = df.loc[next_idx, "Last Price"]
            return df
    
        import pandas as pd
        from pandas.tseries.offsets import BDay
        from xbbg import blp


        #dr = pd.date_range(start=pd.to_datetime(startDate), end=pd.to_datetime(endDate), freq=BDay())
        dr = pd.date_range(start=pd.to_datetime(startDate), end=pd.to_datetime(endDate), freq='D')

        # 2. Mapping dictionary (slot code → time string)
        mapping_dict = {
            "T000": "23:00", "T003": "23:30", "T010": "00:00", "T013": "00:30", "T020": "01:00", "T023": "01:30",
            "T030": "02:00", "T033": "02:30", "T040": "03:00", "T043": "03:30", "T050": "04:00", "T053": "04:30",
            "T060": "05:00", "T063": "05:30", "T070": "06:00", "T073": "06:30", "T080": "07:00", "T083": "07:30",
            "T090": "08:00", "T093": "08:30", "T100": "09:00", "T103": "09:30", "T110": "10:00", "T113": "10:30",
            "T120": "11:00", "T123": "11:30", "T130": "12:00", "T133": "12:30", "T140": "13:00", "T143": "13:30",
            "T150": "14:00", "T153": "14:30", "T160": "15:00", "T163": "15:30", "T170": "16:00", "T173": "16:30",
            "T180": "17:00", "T183": "17:30", "T190": "18:00", "T193": "18:30", "T200": "19:00", "T203": "19:30",
            "T210": "20:00", "T213": "20:30", "T220": "21:00", "T223": "21:30", "T230": "22:00", "T233": "22:30"
        }

        # EFFICIENT APPROACH: Single API call for all tickers
        # print("Fetching data from Bloomberg...")
        ccyStr = f'{ccy}USD' if ccy in ['EUR', 'GBP', 'AUD', 'NZD'] else f'USD{ccy}'
        all_tickers = [f"{ccyStr} {slot_code} Curncy" for slot_code in mapping_dict.keys()]

        # Single bulk API call - much faster!
        bulk_data = blp.bdh(all_tickers, "PX_LAST", startDate, endDate)
        # print(f"Data fetched. Shape: {bulk_data.shape}")

        # Reindex to ensure all business days are present
        bulk_data = bulk_data.reindex(dr)

        # EFFICIENT RESHAPING: Use pandas operations instead of loops
        # print("Reshaping data...")
        # print(f"Bulk data columns: {bulk_data.columns.tolist()[:5]}...")  # Debug: show first 5 columns

        # Handle MultiIndex columns from Bloomberg
        if isinstance(bulk_data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            bulk_data.columns = [col[0] for col in bulk_data.columns]
            # print("Flattened MultiIndex columns")

        # Create a list to store all datetime-price pairs
        data_records = []

        # Vectorized approach using pandas operations
        for slot_code, time_str in mapping_dict.items():
            ticker = f"{ccyStr} {slot_code} Curncy"
            
            if ticker not in bulk_data.columns:
                print(f"Warning: {ticker} not found in data")
                continue
            
            # Get the price series for this time slot
            price_series = bulk_data[ticker]
            
            # Create datetime index by combining date with time
            datetime_index = price_series.index.to_series().dt.strftime('%Y-%m-%d') + ' ' + time_str
            datetime_index = pd.to_datetime(datetime_index)
            
            # Create temporary dataframe with datetime index and prices
            temp_df = pd.DataFrame({
                'datetime': datetime_index,
                'Last Price': price_series.values
            })
            
            # Filter out NaN values
            temp_df = temp_df.dropna()
            
            data_records.append(temp_df)

        # Concatenate all data at once
        # print("Combining all time slots...")
        full_30min_df = pd.concat(data_records, ignore_index=True)

        # Set datetime index and sort
        full_30min_df = full_30min_df.set_index('datetime')
        full_30min_df = full_30min_df.sort_index()

        # Remove any duplicate timestamps (keep first occurrence)
        full_30min_df = full_30min_df[~full_30min_df.index.duplicated(keep='first')]

        # print(f"✅ Complete! Time series shape: {full_30min_df.shape}")
        # print(f"Date range: {full_30min_df.index.min()} to {full_30min_df.index.max()}")
        # print(f"Total data points: {len(full_30min_df)}")
        # print("\nFirst few rows:")
        # print(full_30min_df.head(10))
        # print("\nLast few rows:")
        # print(full_30min_df.tail(5))

        return shift_bfix_late_slots(full_30min_df, late_threshold='16:30')


    # def get30minBfixPrice(ccy, startDate, endDate):
    #     def shift_bfix_late_slots(ccy, df, endDate, late_threshold='16:30'):
    #         mapping_dict = {
    #             "T000": "23:00", "T003": "23:30", "T010": "00:00", "T013": "00:30", "T020": "01:00", "T023": "01:30",
    #             "T030": "02:00", "T033": "02:30", "T040": "03:00", "T043": "03:30", "T050": "04:00", "T053": "04:30",
    #             "T060": "05:00", "T063": "05:30", "T070": "06:00", "T073": "06:30", "T080": "07:00", "T083": "07:30",
    #             "T090": "08:00", "T093": "08:30", "T100": "09:00", "T103": "09:30", "T110": "10:00", "T113": "10:30",
    #             "T120": "11:00", "T123": "11:30", "T130": "12:00", "T133": "12:30", "T140": "13:00", "T143": "13:30",
    #             "T150": "14:00", "T153": "14:30", "T160": "15:00", "T163": "15:30", "T170": "16:00", "T173": "16:30",
    #             "T180": "17:00", "T183": "17:30", "T190": "18:00", "T193": "18:30", "T200": "19:00", "T203": "19:30",
    #             "T210": "20:00", "T213": "20:30", "T220": "21:00", "T223": "21:30", "T230": "22:00", "T233": "22:30"
    #         }

    #         df = df.copy()
    #         df = df.sort_index()
    #         df.index = pd.to_datetime(df.index)
    #         late_slots = [k for k, v in mapping_dict.items() if v >= late_threshold]
    #         late_times = [mapping_dict[slot] for slot in late_slots]
    #         # print(late_times)
    #         all_dates = sorted({d.date() for d in df.index})
    #         for i, d in enumerate(all_dates[:-1]):
    #             today = pd.Timestamp(d)
    #             next_day = pd.Timestamp(all_dates[i+1])
    #             for t in late_times:
    #                 this_idx = pd.Timestamp(f"{today} {t}")
    #                 next_idx = pd.Timestamp(f"{next_day} {t}")
                    

    #                 # get current slot 
    #                 currentSlot = next((k for k, v in mapping_dict.items() if v == t), None)
    #                 ccyStr = f'{ccy}USD' if ccy in ['EUR', 'GBP', 'AUD', 'NZD'] else f'USD{ccy}'
    #                 currentFull = f"{ccyStr} {currentSlot} Curncy"
    #                 currentRate = blp.bdh(currentFull, "PX_LAST", this_idx.strftime('%Y-%m-%d'), next_idx.strftime('%Y-%m-%d'))
    #                 prevRate = blp.bdh(currentFull, "PX_LAST", (this_idx - BDay(1)).strftime('%Y-%m-%d'), this_idx.strftime('%Y-%m-%d'))

    #                 if currentRate.index[0].strftime('%Y-%m-%d') != this_idx.strftime('%Y-%m-%d'):
    #                     # print(f'Date not matched for : {currentRate.index[0].strftime("%Y-%m-%d")}, {this_idx.strftime("%Y-%m-%d")}')
    #                     df.loc[this_idx, "Last Price"] = np.nan
    #                     continue
    #                 if prevRate.shape[0] > 1:
    #                     if prevRate.iloc[1, 0] == currentRate.iloc[0, 0]:
    #                         # print(f'Date not matched for : {currentRate.index[0].strftime("%Y-%m-%d")}, {this_idx.strftime("%Y-%m-%d")}')
    #                         df.loc[this_idx, "Last Price"] = np.nan
    #                         continue

    #                 if (next_idx in df.index) and (this_idx in df.index):
    #                     df.loc[this_idx, "Last Price"] = df.loc[next_idx, "Last Price"]

    #         df = df[df.index < endDate]
    #         # sort the data by date and time
    #         df = df.sort_index()
    #         return df

    #     import pandas as pd
    #     from pandas.tseries.offsets import BDay
    #     from xbbg import blp


    #     #dr = pd.date_range(start=pd.to_datetime(startDate), end=pd.to_datetime(endDate), freq=BDay())
    #     endDate = pd.to_datetime(endDate) + BDay(1) # Include one extra business day 
    #     dr = pd.date_range(start=pd.to_datetime(startDate), end=pd.to_datetime(endDate), freq='D')

    #     # 2. Mapping dictionary (slot code → time string)
    #     mapping_dict = {
    #         "T000": "23:00", "T003": "23:30", "T010": "00:00", "T013": "00:30", "T020": "01:00", "T023": "01:30",
    #         "T030": "02:00", "T033": "02:30", "T040": "03:00", "T043": "03:30", "T050": "04:00", "T053": "04:30",
    #         "T060": "05:00", "T063": "05:30", "T070": "06:00", "T073": "06:30", "T080": "07:00", "T083": "07:30",
    #         "T090": "08:00", "T093": "08:30", "T100": "09:00", "T103": "09:30", "T110": "10:00", "T113": "10:30",
    #         "T120": "11:00", "T123": "11:30", "T130": "12:00", "T133": "12:30", "T140": "13:00", "T143": "13:30",
    #         "T150": "14:00", "T153": "14:30", "T160": "15:00", "T163": "15:30", "T170": "16:00", "T173": "16:30",
    #         "T180": "17:00", "T183": "17:30", "T190": "18:00", "T193": "18:30", "T200": "19:00", "T203": "19:30",
    #         "T210": "20:00", "T213": "20:30", "T220": "21:00", "T223": "21:30", "T230": "22:00", "T233": "22:30"
    #     }

    #     # EFFICIENT APPROACH: Single API call for all tickers
    #     # print("Fetching data from Bloomberg...")
    #     ccyStr = f'{ccy}USD' if ccy in ['EUR', 'GBP', 'AUD', 'NZD'] else f'USD{ccy}'
    #     all_tickers = [f"{ccyStr} {slot_code} Curncy" for slot_code in mapping_dict.keys()]

    #     # Single bulk API call - much faster!
    #     bulk_data = blp.bdh(all_tickers, "PX_LAST", startDate, endDate)
    #     # print(f"Data fetched. Shape: {bulk_data.shape}")

    #     # Reindex to ensure all business days are present
    #     bulk_data = bulk_data.reindex(dr)

    #     # EFFICIENT RESHAPING: Use pandas operations instead of loops
    #     # print("Reshaping data...")
    #     # print(f"Bulk data columns: {bulk_data.columns.tolist()[:5]}...")  # Debug: show first 5 columns

    #     # Handle MultiIndex columns from Bloomberg
    #     if isinstance(bulk_data.columns, pd.MultiIndex):
    #         # Flatten MultiIndex columns
    #         bulk_data.columns = [col[0] for col in bulk_data.columns]
    #         # print("Flattened MultiIndex columns")

    #     # Create a list to store all datetime-price pairs
    #     data_records = []

    #     # Vectorized approach using pandas operations
    #     for slot_code, time_str in mapping_dict.items():
    #         ticker = f"{ccyStr} {slot_code} Curncy"
            
    #         if ticker not in bulk_data.columns:
    #             print(f"Warning: {ticker} not found in data")
    #             continue
            
    #         # Get the price series for this time slot
    #         price_series = bulk_data[ticker]
            
    #         # Create datetime index by combining date with time
    #         datetime_index = price_series.index.to_series().dt.strftime('%Y-%m-%d') + ' ' + time_str
    #         datetime_index = pd.to_datetime(datetime_index)
            
    #         # Create temporary dataframe with datetime index and prices
    #         temp_df = pd.DataFrame({
    #             'datetime': datetime_index,
    #             'Last Price': price_series.values
    #         })
            
    #         # Filter out NaN values
    #         temp_df = temp_df.dropna()
            
    #         data_records.append(temp_df)

    #     # Concatenate all data at once
    #     # print("Combining all time slots...")
    #     full_30min_df = pd.concat(data_records, ignore_index=True)

    #     # Set datetime index and sort
    #     full_30min_df = full_30min_df.set_index('datetime')
    #     full_30min_df = full_30min_df.sort_index()

    #     # Remove any duplicate timestamps (keep first occurrence)
    #     full_30min_df = full_30min_df[~full_30min_df.index.duplicated(keep='first')]

    #     # print(f"✅ Complete! Time series shape: {full_30min_df.shape}")
    #     # print(f"Date range: {full_30min_df.index.min()} to {full_30min_df.index.max()}")
    #     # print(f"Total data points: {len(full_30min_df)}")
    #     # print("\nFirst few rows:")
    #     # print(full_30min_df.head(10))
    #     # print("\nLast few rows:")
    #     # print(full_30min_df.tail(5))

    #     return shift_bfix_late_slots(ccy, full_30min_df, endDate, late_threshold='16:30')

    def fetchData(self, startDate, endDate):
        from xbbg import blp
        dr = pd.date_range(
            start=pd.to_datetime(startDate),
            end=pd.to_datetime(endDate),
            freq=BDay()
        )

        # get vol data
        self.vol = blp.bdh(self.volCode, 'PX_LAST', start_date=startDate, end_date=endDate, Per='D').iloc[:, 0].reindex(dr).ffill()

        drLibor = pd.date_range(
            start=pd.to_datetime('2010-01-01'),
            end=pd.to_datetime('2024-09-30'),
            freq=BDay()
        )

        drSofr = pd.date_range(
            start=pd.to_datetime('2024-10-01'),
            end=pd.to_datetime(endDate),
            freq=BDay()
        )



        liborUSD = pd.DataFrame(index=drLibor)
        sofrUSD = pd.DataFrame(index=drSofr)

        liborUSD['libor'] = blp.bdh('US0001M Index', 'PX_LAST', start_date=drLibor[0], end_date=drLibor[-1], Per='D').iloc[:, 0]
        sofrUSD['sofr'] = blp.bdh('SOFR30A Index', 'PX_LAST', start_date=drSofr[0], end_date=drSofr[-1], Per='D').iloc[:, 0] # prev is SOFRRATE
        # concat liborUSD and sofrUSD which is in different time period
        liborUSD = liborUSD[~liborUSD.index.duplicated(keep='first')]
        sofrUSD = sofrUSD[~sofrUSD.index.duplicated(keep='first')]
        # Merge the two series into a continuous series
        usdRate = liborUSD['libor'].combine_first(sofrUSD['sofr'])
        # Sort the index to ensure a continuous time series
        usdRate = usdRate.sort_index()
        self.usdRate = usdRate.reindex(dr).ffill()
        
        # if self.ccyLeft == 'USD':
        #     self.leftRate = usdRate.reindex(dr).ffill()
        #     # self.rightRate = blp.bdh(self.rightRateCode, 'PX_LAST', start_date=startDate, end_date=endDate, Per='D').iloc[:, 0].reindex(dr).ffill()
        # elif self.ccyRight == 'USD':
        #     # self.leftRate = blp.bdh(self.leftRateCode, 'PX_LAST', start_date=startDate, end_date=endDate, Per='D').iloc[:, 0].reindex(dr).ffill()
        #     self.rightRate = usdRate.reindex(dr).ffill()

        # use BGN price instead of CMTK
        self.spot = blp.bdh(f'USD{self.ccy} BGN Curncy' if self.ccy not in self.revL else f'{self.ccy}USD BGN Curncy', 'PX_LAST', start_date=startDate, end_date=endDate, Per='D').iloc[:, 0].reindex(dr).ffill()
        self.spotBfixTokyo = blp.bdh(f'{self.ccyFull} T150 Curncy', "PX_LAST", startDate, endDate).iloc[1:, 0].reindex(dr).ffill() # don't need to shift for one day but delete first date to align with other data
        
        swap = blp.bdh(self.swapCode, 'PX_LAST', start_date=startDate, end_date=endDate, Per='D').iloc[:, 0].reindex(dr).ffill()
        # change swap to pips
        self.swap = swap * 0.0001 if self.ccy != 'JPY' else swap * 0.01

        # shift all the data for one day
        self.vol = self.vol.shift(1)
        # self.leftRate = self.leftRate.shift(1)
        # self.rightRate = self.rightRate.shift(1)
        self.spot = self.spot.shift(1)
        self.swap = self.swap.shift(1)

        # remove the first row which is NaN
        self.vol = self.vol.dropna()
        # self.leftRate = self.leftRate.dropna()
        # self.rightRate = self.rightRate.dropna()
        self.spot = self.spot.dropna()
        self.swap = self.swap.dropna()


        self.fwd = self.spotBfixTokyo + self.swap
        

    def getOptionPrice(self, s, K, F, sigma, expire, delivery, usdRate, foredayBasis, cptydayBasis, strgy):
        # getOptionPrice(self, s, K, sigma, expire, delivery, foreRate, cptyRate, foredayBasis, cptydayBasis, strgy)

        # self.foredayBasis = self.usdDayCount if self.ccyLeft == 'USD' else self.otherCcyDayCount
        # self.cptydayBasis = self.otherCcyDayCount if self.ccyLeft == 'USD' else self.usdDayCount
        self.foredayBasis = foredayBasis
        self.cptydayBasis = cptydayBasis
        

        base = 360
        t = expire / base
        
        # FIX: Use proper time fraction for rate calculation (annual basis)
        T_annual = expire / base  # Use standard annual fraction
        
    
        fwd_points_1m = F - s
        fwd_points_1w = fwd_points_1m * (delivery / 30)  # Assuming F is 1M forward
        F_new = s + fwd_points_1w
        # Use market forward directly
        # F_new = F
        
        if self.ccyLeft == 'USD':  # USD/XXX pair
            # CORRECTED: Use annual time fraction, not money market fraction
            impliedCcyRate = usdRate + np.log(F_new / s) / T_annual
            foreRate = usdRate
            cptyRate = impliedCcyRate
            
        elif self.ccyRight == 'USD':  # XXX/USD pair
            # CORRECTED: Use annual time fraction
            impliedCcyRate = usdRate - np.log(F_new / s) / T_annual
            foreRate = impliedCcyRate
            cptyRate = usdRate
        

        if delivery > self.cptydayBasis:
            CtrYield = base / expire * (delivery / self.cptydayBasis) * math.log(1 + cptyRate) # cpty 就是右邊rate

        else:
            CtrYield = base / expire * math.log(1 + cptyRate * delivery / self.cptydayBasis)

        if delivery > self.foredayBasis:
            Yield = base / expire * (delivery / self.foredayBasis) * math.log(1 + foreRate) # fore 就是左邊rate

        else:
            Yield = base / expire * math.log(1 + foreRate * delivery / self.foredayBasis)
    

        # # identify which rate is USD
        # if self.ccyFull[:3] == 'USD': # USDXXX
        #     self.usdRate = self.leftRate
        # else: # XXXUSD
        #     self.usdRate = self.rightRate
        # self.impliedCcyRate = self.usdRate - (1.0 / t) * np.log(self.fwd / self.spot)
        
        d1 = (math.log(s / K) + ((CtrYield - Yield) + 0.5 * sigma * sigma) * t) / sigma / math.sqrt(t)
        d2 = d1 - sigma * math.sqrt(t)


        if strgy == "Call":
            delta = math.exp(-Yield * t) * norm.cdf(d1)
            european_option = s * delta - K * math.exp(-CtrYield * t) * norm.cdf(d2)
            # print all paramters to inspect:
            print(f's: {s}, K: {K}, F_new: {F_new}, sigma: {sigma}, expire: {expire}, delivery: {delivery}, usdRate: {usdRate}, foredayBasis: {foredayBasis}, cptydayBasis: {cptydayBasis}, strgy: {strgy}')
            print(f'option Price: {european_option}')

            return european_option 
        
        elif strgy == "Put":
            delta = -math.exp(-Yield * t) * norm.cdf(-d1)
            european_option = K * math.exp(-CtrYield * t) * norm.cdf(-d2) + s * delta
           
            print(f's: {s}, K: {K}, F_new: {F_new}, sigma: {sigma}, expire: {expire}, delivery: {delivery}, usdRate: {usdRate}, foredayBasis: {foredayBasis}, cptydayBasis: {cptydayBasis}, strgy: {strgy}')
            print(f'option Price: {european_option}')

            return european_option 
        else:
            print("error option strategy")
            return None
    

    def getOptionPrice_Percent(self, s, K, F, sigma, expire, delivery, usdRate, foredayBasis, cptydayBasis, strgy):
        self.foredayBasis = foredayBasis
        self.cptydayBasis = cptydayBasis
        

        base = 360
        t = expire / base
        
        # FIX: Use proper time fraction for rate calculation (annual basis)
        T_annual = expire / base  # Use standard annual fraction
        
    
        fwd_points_1m = F - s
        fwd_points_1w = fwd_points_1m * (delivery / 30)  # Assuming F is 1M forward
        F_new = s + fwd_points_1w
        # Use market forward directly
        # F_new = F
        
        if self.ccyLeft == 'USD':  # USD/XXX pair
            # CORRECTED: Use annual time fraction, not money market fraction
            impliedCcyRate = usdRate + np.log(F_new / s) / T_annual
            foreRate = usdRate
            cptyRate = impliedCcyRate
            
        elif self.ccyRight == 'USD':  # XXX/USD pair
            # CORRECTED: Use annual time fraction
            impliedCcyRate = usdRate - np.log(F_new / s) / T_annual
            foreRate = impliedCcyRate
            cptyRate = usdRate
        

        if delivery > self.cptydayBasis:
            CtrYield = base / expire * (delivery / self.cptydayBasis) * math.log(1 + cptyRate) # cpty 就是右邊rate

        else:
            CtrYield = base / expire * math.log(1 + cptyRate * delivery / self.cptydayBasis)

        if delivery > self.foredayBasis:
            Yield = base / expire * (delivery / self.foredayBasis) * math.log(1 + foreRate) # fore 就是左邊rate

        else:
            Yield = base / expire * math.log(1 + foreRate * delivery / self.foredayBasis)
    

        
        d1 = (math.log(s / K) + ((CtrYield - Yield) + 0.5 * sigma * sigma) * t) / sigma / math.sqrt(t)
        d2 = d1 - sigma * math.sqrt(t)


        if strgy == "Call":
            delta = math.exp(-Yield * t) * norm.cdf(d1)
            european_option = s * delta - K * math.exp(-CtrYield * t) * norm.cdf(d2)
            # print all paramters to inspect:
            print(f's: {s}, K: {K}, F_new: {F_new}, sigma: {sigma}, expire: {expire}, delivery: {delivery}, usdRate: {usdRate}, foredayBasis: {foredayBasis}, cptydayBasis: {cptydayBasis}, strgy: {strgy}')
            print(f'option Price: {european_option/s}')

            return european_option / s # convert to percentage
        
        elif strgy == "Put":
            delta = -math.exp(-Yield * t) * norm.cdf(-d1)
            european_option = K * math.exp(-CtrYield * t) * norm.cdf(-d2) + s * delta
           
            print(f's: {s}, K: {K}, F_new: {F_new}, sigma: {sigma}, expire: {expire}, delivery: {delivery}, usdRate: {usdRate}, foredayBasis: {foredayBasis}, cptydayBasis: {cptydayBasis}, strgy: {strgy}')
            print(f'option Price: {european_option/s}')

            return european_option  / s
        else:
            print("error option strategy")
            return None

    # def getHisVol(self, ccy, startDate, endDate, n):
    #     from xbbg import blp
    #     dr = pd.date_range(
    #         start=pd.to_datetime(startDate),
    #         end=pd.to_datetime(endDate),
    #         freq=BDay()
    #     )
    #     spot = blp.bdh(f'USD{ccy} BGN Curncy' if ccy not in self.revL else f'{ccy}USD BGN Curncy', 'PX_LAST', start_date=startDate, end_date=endDate, Per='D').iloc[:, 0].reindex(dr).ffill()
    #     spotRet = spot.pct_change().dropna()
    #     # get volatility from spot return
    #     rollingVol = spotRet.rolling(window=n).std() * np.sqrt(252)
    #     rollingVol = rollingVol.reindex(dr).ffill()

    #     return rollingVol



    # def getDeltaStrike(self,
    #     ccyFull, s, F, sigma, expire, delivery, usdRate, foredayBasis, cptydayBasis,
    #     target_delta, option_type
    # ):
    #     """
    #     Calculate strike K for given option delta (FX Garman-Kohlhagen, Bloomberg premium-adjusted convention).
    #     Inputs:
    #         ccyFull: currency pair string, e.g. 'EURUSD'
    #         s: spot price
    #         F: forward price (1M)
    #         sigma: volatility (annualized, e.g. 0.10 for 10%)
    #         expire: option expiry (days)
    #         delivery: option delivery (days)
    #         usdRate: USD interest rate (as decimal, e.g. 0.04 for 4%)
    #         foredayBasis: day basis for foreign currency
    #         cptydayBasis: day basis for counterparty currency
    #         target_delta: desired premium-adjusted delta (e.g. 0.25 for 25D call, -0.25 for 25D put)
    #         option_type: "Call" or "Put"
    #     Returns:
    #         K: Strike price corresponding to desired premium-adjusted delta (Bloomberg)
    #     """
    #     from scipy.stats import norm
    #     from scipy.optimize import brentq
    #     base = 360
    #     t = expire / base
    #     T_annual = expire / base

    #     # Calculate forward adjustment
    #     fwd_points_1m = F - s
    #     fwd_points_1w = fwd_points_1m * (delivery / 30)
    #     F_new = s + fwd_points_1w
    #     # Use market forward directly if preferred
    #     # F_new = F

    #     ccyLeft = ccyFull[:3]
    #     ccyRight = ccyFull[3:]

    #     # Calculate yields
    #     if ccyLeft == 'USD':
    #         impliedCcyRate = usdRate + np.log(F_new / s) / T_annual
    #         foreRate = usdRate
    #         cptyRate = impliedCcyRate
    #     elif ccyRight == 'USD':
    #         impliedCcyRate = usdRate - np.log(F_new / s) / T_annual
    #         foreRate = impliedCcyRate
    #         cptyRate = usdRate
    #     else:
    #         impliedCcyRate = usdRate
    #         foreRate = usdRate
    #         cptyRate = usdRate

    #     # Yield conventions
    #     if delivery > cptydayBasis:
    #         CtrYield = base / expire * (delivery / cptydayBasis) * np.log(1 + cptyRate)
    #     else:
    #         CtrYield = base / expire * np.log(1 + cptyRate * delivery / cptydayBasis)
    #     if delivery > foredayBasis:
    #         Yield = base / expire * (delivery / foredayBasis) * np.log(1 + foreRate)
    #     else:
    #         Yield = base / expire * np.log(1 + foreRate * delivery / foredayBasis)

    #     r_dom = Yield      # domestic
    #     r_for = CtrYield   # foreign

    #     df_dom = np.exp(-r_dom * t)
    #     df_for = np.exp(-r_for * t)
    #     sqrt_t = np.sqrt(t)

    #     def black_price(K):
    #         d1 = (np.log(s / K) + (r_dom - r_for + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    #         d2 = d1 - sigma * sqrt_t
    #         if option_type == "Call":
    #             return s * df_for * norm.cdf(d1) - K * df_dom * norm.cdf(d2)
    #         elif option_type == "Put":
    #             return K * df_dom * norm.cdf(-d2) - s * df_for * norm.cdf(-d1)
    #         else:
    #             raise ValueError("option_type must be 'Call' or 'Put'")

    #     def pa_delta(K):
    #         d1 = (np.log(s / K) + (r_dom - r_for + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    #         P = black_price(K)
    #         if option_type == "Call":
    #             return df_for * norm.cdf(d1) - P / s
    #         elif option_type == "Put":
    #             return -df_for * norm.cdf(-d1) + P / s
    #         else:
    #             raise ValueError("option_type must be 'Call' or 'Put'")

    #     # Root search for K such that pa_delta(K) == target_delta (Bloomberg convention)
    #     K_min = s * 0.5
    #     K_max = s * 1.5
    #     K = brentq(lambda K: pa_delta(K) - target_delta, K_min, K_max)
    #     return K
    
    # from scipy.stats import norm
    def getDeltaStrike(self, ccyFull, s, F, sigma, expire, delivery, usdRate, foredayBasis, cptydayBasis, target_delta, option_type):
        """
        Calculate strike K for given option delta (FX Garman-Kohlhagen, Bloomberg convention).
        Inputs:
            ccyFull: currency pair string, e.g. 'EURUSD'
            s: spot price
            F: forward price (1M)
            sigma: volatility (annualized, e.g. 0.10 for 10%)
            expire: option expiry (days)
            delivery: option delivery (days)
            usdRate: USD interest rate
            foredayBasis: day basis for foreign currency
            cptydayBasis: day basis for counterparty currency
            target_delta: desired delta (e.g. 0.25 for 25D call, -0.25 for 25D put)
            option_type: "Call" or "Put"
        Returns:
            K: Strike price corresponding to desired delta
        """
        from scipy.stats import norm

        ccyLeft = ccyFull[:3]  # Left currency (e.g. USD)
        ccyRight = ccyFull[3:]  # Right currency (e.g. EUR)

        base = 365
        t = expire / base
        T_annual = expire / base

        # fwd_points_1m = F - s
        # fwd_points_1w = fwd_points_1m * (delivery / 30)
        # F_new = s + fwd_points_1w
        # Use market forward directly if preferred
        F_new = F

        # Calculate yields
        if ccyLeft == 'USD': # USDXXX
            impliedCcyRate = usdRate + np.log(F_new / s) / T_annual
            foreRate = usdRate
            cptyRate = impliedCcyRate
        elif ccyRight == 'USD': # XXXUSD
            impliedCcyRate = usdRate - np.log(F_new / s) / T_annual
            foreRate = impliedCcyRate
            cptyRate = usdRate
        else:
            # If neither side is USD, default to Bloomberg convention (you may want to amend for your needs)
            impliedCcyRate = usdRate
            foreRate = usdRate
            cptyRate = usdRate

        if delivery > cptydayBasis:
            CtrYield = base / expire * (delivery / cptydayBasis) * np.log(1 + cptyRate)
        else:
            CtrYield = base / expire * np.log(1 + cptyRate * delivery / cptydayBasis)
        if delivery > foredayBasis:
            Yield = base / expire * (delivery / foredayBasis) * np.log(1 + foreRate)
        else:
            Yield = base / expire * np.log(1 + foreRate * delivery / foredayBasis)

        # # Discount factor for yield
        # discount = np.exp(-Yield * t)
        # if option_type == "Call":
        #     target = target_delta / discount
        #     d1 = norm.ppf(target)
        # elif option_type == "Put":
        #     target = -target_delta / discount
        #     d1 = -norm.ppf(target)
        #     print(target, d1)
        # else:
        #     raise ValueError("option_type must be 'Call' or 'Put'")

        discount = np.exp(-Yield * t) # yield 代表的是foreign rate
        if option_type == "Call":
            target = target_delta / discount
            # Clamp to (epsilon, 1-epsilon)
            target = min(max(target, 1e-8), 1 - 1e-8)
            d1 = norm.ppf(target)
        elif option_type == "Put":
            target = -target_delta / discount
            target = min(max(target, 1e-8), 1 - 1e-8)
            d1 = -norm.ppf(target)
        else:
            raise ValueError("option_type must be 'Call' or 'Put'")

        # Solve for K
        K = s * np.exp(-sigma * np.sqrt(t) * d1 + (CtrYield - Yield + 0.5 * sigma ** 2) * t)
        return K
    
    def optionProfit_straddle(self, dir, callPx, putPx, K=None, fixingRate=None):
        dir = dir.upper()
        prem = callPx + putPx
        if fixingRate and K:
            fixKProfit = abs(fixingRate-K) # 對straddle來說一定有一方獲利
            netProfit = fixKProfit - prem
        if dir == 'SELL':
            prem = -1 * prem
            netProfit = -1 * netProfit
        return prem, netProfit

    def getOptionPrice_strangle(self, dir, highCallPx, lowPutPx, highCallK=None, lowPutK=None, fixingRate=None):
        dir = dir.upper()
        prem = highCallPx + lowPutPx
        if fixingRate and highCallK and lowPutK:
            if fixingRate >= highCallK: # call 執行但put不執行
                fixKProfit = fixingRate - highCallK
            elif lowPutK < fixingRate < highCallK: # call, put均不執行
                fixKProfit = 0
            elif fixingRate <= lowPutK: # put 執行但call不執行
                fixKProfit = lowPutK - fixingRate
            netProfit = fixKProfit - prem 
        if dir == 'SELL':
            prem = -1 * prem
            netProfit = -1 * netProfit
        return prem, netProfit
    
    def getOptionPrice_rr(self, dir, highCallPx, lowPutPx, highCallK=None, lowPutK=None, fixingRate=None):
        dir = dir.upper()
        prem = highCallPx - lowPutPx
        if fixingRate and highCallK and lowPutK:
            if fixingRate >= highCallK:
                fixKProfit = fixingRate - highCallK
            elif lowPutK < fixingRate < highCallK:
                fixKProfit = 0
            elif fixingRate <= lowPutK:
                fixKProfit = lowPutK - fixingRate
            netProfit = fixKProfit - prem

        if dir == 'SELL':
            prem = -1 * prem
            netProfit = -1 * netProfit
        return prem, netProfit

    def getOptionPrice_spread(self, dir, use, highPx, lowPx, highK=None, lowK=None, fixingRate=None):
        dir = dir.upper()
        use = use.upper()
        
        if use == 'CALL': # buy low K call and sell high K call
            prem = higPx - highPx

        # extra monitr 
        if dir == 'BUY':
            return highPx - lowPx
        if dir == 'SELL':
            return lowPx - highPx