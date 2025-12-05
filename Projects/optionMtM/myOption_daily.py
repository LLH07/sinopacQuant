import numpy as np
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(r"C:\Users\Blair Lin\quantTrade_2025\optionDailyMtM\data_測試")
from datetime import datetime

# Standard normal PDF and CDF
N_pdf = stats.norm.pdf
N = stats.norm.cdf

class option():
    def __init__(self, CCY, endDate):
        self.CCY = CCY
        self.ccyLeft = CCY[:3]
        self.ccyRight = CCY[3:]
        self.dayCount365 = ['EURUSD', 'GBPUSD', 'AUDNZD', 'NZDUSD', 'USDZAR', 'USDCAD']
        self.today = datetime.today().strftime('%Y-%m-%d')
        self.ccyList = [CCY]
        self.endDate = endDate        

        # fetch revlevant data
        print(f'抓取 {self.CCY} Option參數中，截止日期: {self.endDate} ......')
        self.vol = self.fetchVolTable()
        self.volExpDate = self.fetchVolExpDate()
        self.fwd = self.fetchFwd()
        self.fwdDate = self.fetchFwdDate()
        self.spot = self.fetchSpot()
        self.spotDate = self.fetchSpotDate()
        self.usdRate = self.fetchUSDRate()
        self.rateDate = self.fetchRateDate()
        print(f'資料抓取完成！')

    def fetchVolTable(self) -> pd.DataFrame:
        import numpy as np
        import pandas as pd
        from datetime import datetime
        from xbbg import blp

        tenor = ["ON", "1W", "2W", "3W", "1M", "2M", "3M", "4M", "6M", "9M", "1Y"]
        def vol_download(currency: str) -> pd.DataFrame:
            option_type = ["V", "25R", "25B", "10R", "10B"]

            # build tickers in the same order as the original code (option_type outer, tenor inner)
            vol_ticker = [currency + j + i + " BGN Curncy" for j in option_type for i in tenor]
            # arrays describing the MultiIndex for columns (must match vol_ticker order)
            vol_column_tenor = [i for j in option_type for i in tenor]
            vol_column_type = [j for j in option_type for i in tenor]

            # today (end date)
            today = datetime.now().strftime("%Y-%m-%d")

            # fetch time series: one field -> DataFrame with columns == tickers
            vol = blp.bdh(tickers=vol_ticker, flds=['PX_LAST'], start_date='2010-01-01', end_date=self.endDate, Per='D', Calendar='5D')
            vol.index = pd.DatetimeIndex(vol.index)

            # forward-fill missing values
            vol.fillna(method='ffill', inplace=True)

            # convert columns to a MultiIndex (tenor, option_type)
            vol.columns = pd.MultiIndex.from_arrays([vol_column_tenor, vol_column_type], names=["tenor", "type"])

            # add currency as first level of the index (so concatenation across currencies is straightforward)
            vol_index_currency = [currency] * len(vol)
            vol.index = pd.MultiIndex.from_arrays([vol_index_currency, vol.index], names=["currency", "date"])

            return vol


        # Build the combined VOL dataframe for all currencies in CCY_LIST
        VOL = pd.DataFrame()

        # Ensure CCY_LIST is defined externally; error if not.
        try:
            self.ccyList
        except NameError:
            raise NameError("CCY_LIST is not defined. Define CCY_LIST = ['EURUSD', ...] before running this script.")

        for currency in self.ccyList:
            vol = vol_download(currency)
            VOL = pd.concat([VOL, vol], axis=0)

        # VOL has columns MultiIndex (tenor, type). Stack the tenor level so index becomes (currency, date, tenor)
        VOL = VOL.stack(level=0)  # now index = (currency, date, tenor) and columns = option types like 'V','25R',...

        # Ensure column names we expect exist; if not, they'll be treated as NaN
        # Compute standard interpolated option columns (25C, 25P, 10C, 10P)
        # Use .astype(float) to ensure numeric operations
        for col in ["V", "25R", "25B", "10R", "10B"]:
            if col not in VOL.columns:
                VOL[col] = np.nan
        VOL = VOL.astype(float)

        VOL["25C"] = (VOL["25B"] + VOL["V"]) + (VOL["25R"] / 2)
        VOL["25P"] = (VOL["25B"] + VOL["V"]) - (VOL["25R"] / 2)
        VOL["10C"] = (VOL["10B"] + VOL["V"]) + (VOL["10R"] / 2)
        VOL["10P"] = (VOL["10B"] + VOL["V"]) - (VOL["10R"] / 2)

        # Drop the raw inputs we no longer need
        VOL = VOL.drop(["10B", "10R", "25B", "25R"], axis=1, errors='ignore')

        # Reorder columns to the desired structure: 10C,25C,V,25P,10P
        desired_cols = ["10C", "25C", "V", "25P", "10P"]
        # keep only existing desired columns in that order
        # cols_to_keep = [c for c in desired_cols if c in VOL.columns]
        cols_to_keep = desired_cols
        VOL = VOL.reindex(columns=cols_to_keep)

        # Rename columns to percentiles as in original code
        VOL.columns = [0.1, 0.25, 0.5, 0.75, 0.9][: len(VOL.columns)]

        # Reset index into columns (optional) and set index names as requested
        VOL = VOL.reset_index()
        VOL["tenor"] = pd.Categorical(VOL["tenor"], categories=tenor, ordered=True)
        VOL = VOL.sort_values(["currency", "date", "tenor"])
        VOL = VOL.set_index(["currency", "date", "tenor"])
        VOL.index.names = ["currency", "date", "tenor"]

        return VOL


    def fetchVolExpDate(self) -> pd.DataFrame:
        import pandas as pd
        from datetime import datetime
        import os

        # Ensure tenor and CCY_LIST are defined in your environment, or define them here:
        tenor = ["ON", "1W", "2W", "3W", "1M", "2M", "3M", "4M", "6M", "9M", "1Y"]
        # Example: CCY_LIST = ["EURUSD", "USDJPY"]
        # CCY_LIST must be defined before running this script.
        try:
            self.ccyList
        except NameError:
            raise NameError("CCY_LIST is not defined. Define CCY_LIST = ['EURUSD', ...] before running this script.")

        # Path to the Excel file - adjust as needed
        input_path = r"C:\Users\Blair Lin\quantTrade_2025\optionDailyMtM\data_更新版\volExpDate.xlsx"
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} not found")

        # Result DataFrame: columns = tenor, index will be a MultiIndex (currency, date)
        VOL_exercise_date = pd.DataFrame(columns=tenor)

        for currency in self.ccyList:
            # read sheet into DataFrame
            vol_exercise_date = pd.read_excel(input_path, sheet_name=currency)

            # Ensure there is a 'Date' column
            if "Date" not in vol_exercise_date.columns:
                raise KeyError(f"'Date' column not found in sheet '{currency}'")

            # set index from Date column (parse dates robustly)
            # Some Date cells may already be datetimes, others strings 'YYYY/MM/DD'
            vol_exercise_date["Date_parsed"] = pd.to_datetime(vol_exercise_date["Date"], format="%Y/%m/%d", errors="coerce")
            # If parsing with format failed, try generic parse
            mask_na = vol_exercise_date["Date_parsed"].isna()
            if mask_na.any():
                vol_exercise_date.loc[mask_na, "Date_parsed"] = pd.to_datetime(vol_exercise_date.loc[mask_na, "Date"], errors="coerce")

            # drop rows where Date couldn't be parsed
            vol_exercise_date = vol_exercise_date.dropna(subset=["Date_parsed"])

            # keep only tenor columns that exist in the sheet (avoid KeyError)
            present_tenors = [t for t in tenor if t in vol_exercise_date.columns]
            if not present_tenors:
                # nothing to append for this sheet
                continue

            vol_exercise_date = vol_exercise_date.set_index("Date_parsed")[present_tenors]
            # convert index to plain python date objects (if you prefer)
            vol_exercise_date.index = vol_exercise_date.index.date
            vol_exercise_date.index = pd.DatetimeIndex(vol_exercise_date.index)

            # create a MultiIndex with currency as first level and date as second
            multi_index = pd.MultiIndex.from_arrays(
                [[currency] * len(vol_exercise_date), list(vol_exercise_date.index)],
                names=["currency", "date"],
            )
            vol_exercise_date.index = multi_index

            # concat to the master DataFrame
            VOL_exercise_date = pd.concat([VOL_exercise_date, vol_exercise_date], axis=0, sort=False)

        # Optional: ensure final index is a MultiIndex with the expected names
        if not isinstance(VOL_exercise_date.index, pd.MultiIndex):
            # try to interpret existing index as tuples
            try:
                idx0 = [i[0] for i in VOL_exercise_date.index]
                idx1 = [i[1] for i in VOL_exercise_date.index]
                VOL_exercise_date.index = pd.MultiIndex.from_arrays([idx0, idx1], names=["currency", "date"])
            except Exception:
                pass

        VOL_exercise_date.index.names = ["currency", "date"]

        return VOL_exercise_date

    def fetchFwd(self) -> pd.DataFrame:
        from xbbg import blp
        spot = pd.DataFrame(columns = self.ccyList)
        spot_ticker = [i+ " CMTK Curncy" for i in self.ccyList]
        spot = blp.bdh(tickers=spot_ticker , flds=['PX_LAST'] , start_date='2010-01-01', end_date=self.endDate,Per = 'D',Calendar='5D')
        spot.columns = self.ccyList
        spot_diff = spot.diff()
        spot_chg = spot.diff()/spot

        tenor2 = ["ON","1W","2W","3W","1M","2M","3M","4M","6M","9M","12M"]
        def forward_download(currency):
            swap_ticker=[]
            swap_column_1=[]
            swap_column_2=[]
        
            for i in tenor2:
                swap_ticker.append(currency+i+" CMTK Curncy")

            today = datetime.now().strftime("%Y-%m-%d")
            swap = blp.bdh(tickers=swap_ticker , flds=['PX_LAST'] , start_date='2010-01-01', end_date=self.endDate,Per = 'D',Calendar='5D')
            swap.fillna(method='ffill',inplace=True)
            swap.columns = tenor2
        
            scale = blp.bdp(currency +" CURNCY","FWD_SCALE")
            swap = swap / (10**scale.iloc[0,0])
        
            spot_ticker = currency + " CMTK Curncy"
            spot = blp.bdh(tickers=spot_ticker , flds=['PX_LAST'] , start_date='2010-01-01', end_date=self.endDate,Per = 'D',Calendar='5D')
            spot.columns = [currency]
        
            forward = pd.DataFrame()
            for i in swap.columns:
                forward[i] = swap[i] + spot[currency]
            forward.index = pd.DatetimeIndex(forward.index)
            
            forward_index1 = np.array([currency]*len(forward))
            forward_index2 = forward.index
            forward = forward.set_index([forward_index1,forward_index2])
            return forward

        FORWARD = pd.DataFrame(columns = tenor2)
        
        for currency in self.ccyList:
            forward = forward_download(currency)
            FORWARD = pd.concat([FORWARD,forward],axis=0)
        
        
        FORWARD_index1 = [i[0] for i in FORWARD.index]
        FORWARD_index2 = [i[1] for i in FORWARD.index]
        FORWARD= FORWARD.set_index([FORWARD_index1,FORWARD_index2])   

        return FORWARD


    def fetchFwdDate(self):
        Fwd_date = pd.DataFrame()
        tenor2 = ["ON","1W","2W","3W","1M","2M","3M","4M","6M","9M","12M"]
        fwdTenor = tenor2

        for currency in self.ccyList:  
            fwd_date = pd.read_excel(r"C:\Users\Blair Lin\quantTrade_2025\optionDailyMtM\data_更新版\fwdValDate.xlsx",sheet_name=currency)
            fwd_date.index = fwd_date["Date"]
            fwd_date.index = pd.DatetimeIndex(fwd_date.index)

            fwd_date = fwd_date.iloc[1:]
            fwd_date_index1 = [currency] * len(fwd_date)
            fwd_date_index2 = fwd_date.index
            fwd_date = fwd_date.set_index([fwd_date_index1,fwd_date_index2])
        
            Fwd_date = pd.concat([Fwd_date,fwd_date],axis=0)
        
        return Fwd_date


    def fetchSpot(self):
        from xbbg import blp
        def spot_download(currency):
            spot_ticker=[]

            spot_ticker.append(currency+" CMTK Curncy")
            today = datetime.now().strftime("%Y-%m-%d")
        
            spot = blp.bdh(tickers=spot_ticker , flds=['PX_LAST'] , start_date='2010-01-01', end_date=self.endDate,Per = 'D',Calendar='5D')
            # change to datetime index
            spot.index = pd.DatetimeIndex(spot.index)
            # spot.columns = [currency]

            return spot

        SPOT = pd.DataFrame(columns = self.ccyList)
        for currency in self.ccyList:
            spot = spot_download(currency)
            SPOT[currency] = spot.iloc[:, 0]
        
        return SPOT


    def fetchSpotDate(self):
        spot_date = pd.read_excel(rf"C:\Users\Blair Lin\quantTrade_2025\optionDailyMtM\data_更新版\spotValDate_new.xlsx")
        spot_date.index = spot_date["Date"]
        spot_date.index = pd.DatetimeIndex(spot_date.index)
        spot_date = spot_date.iloc[1:,6:]

        return spot_date


    def fetchUSDRate(self):
        from xbbg import blp
        rate_ticker = ['USOSFR1Z Curncy', 'USOSFR2Z Curncy', 'USOSFRA Curncy', 'USOSFRB Curncy', 'USOSFRC Curncy', 'USOSFRD Curncy', 'USOSFRF Curncy', 'USOSFRI Curncy', 'USOSFR1 Curncy']
        tenor_USD = ['1W', '2W', '1M', '2M', '3M', '4M', '6M', '9M', '12M']

        rate = blp.bdh(tickers=rate_ticker , flds=['PX_LAST'] , start_date='2010-01-01', end_date=self.endDate,Per = 'D',Calendar='5D')
        rate.index = pd.DatetimeIndex(rate.index)
        rate.fillna(method='ffill',inplace=True)
        rate.columns = tenor_USD

        rate_index1 = np.array(["USD"]*len(rate))
        rate_index2 = [i for i in rate.index]
        rate = rate.set_index([rate_index1,rate_index2])

        Rate = pd.DataFrame()
        Rate = pd.concat([Rate,rate],axis=0)
        
        return Rate

    def fetchRateDate(self):
        Rate_date = pd.DataFrame()
        tenor = ["1W","2W","1M","2M","3M","4M","6M","9M","1Y"]

        for currency in self.ccyList:  
            rate_date = pd.read_excel(r"C:\Users\Blair Lin\quantTrade_2025\optionDailyMtM\data_更新版\fwdValDate.xlsx",sheet_name=currency)
            rate_date.index = rate_date["Date"]
            rate_date.index = pd.DatetimeIndex(rate_date.index)
        
            rate_date = rate_date[tenor]
            rate_date = rate_date.iloc[1:]
            #rate_date = rate_date.applymap(lambda x : datetime.strptime(x, '%Y/%m/%d'))
            rate_date_index1 = [currency] * len(rate_date)
            # rate_date_index2 = [i.date() for i in rate_date.index]
            rate_date_index2 = rate_date.index
            rate_date = rate_date.set_index([rate_date_index1,rate_date_index2])
        
            Rate_date = pd.concat([Rate_date,rate_date],axis=0)
        
        return Rate_date




    def create_forward_curve(self, market_tenors_days, market_forward_prices):
        """
        Creates an interpolation function for the Forward Price curve.

        Args:
            market_tenors_days (list/array): Days to expiry (including 0 for spot).
            market_forward_prices (list/array): Outright forward prices (including spot).

        Returns:
            scipy.interpolate.interp1d: A callable function to get F(T).
        """
        import numpy as np
        from scipy.interpolate import interp1d
        # 1. Convert days to years
        T_years = np.array(market_tenors_days) / 360
        F_prices = np.array(market_forward_prices)
        
        # 2. Choose the interpolation method
        # 'linear' is the simplest, 'cubic' or 'quadratic' are smoother but riskier at endpoints.
        # 'linear' interpolation is common practice for Forward Prices (F).
        forward_curve_func = interp1d(
            T_years, 
            F_prices, 
            kind='linear', 
            fill_value="extrapolate" # Allows estimation outside the known range
        )
        
        return forward_curve_func

    def Imply_K_call_F(self, guess, S, F, rd, sigma, t, delta, tolerance=1e-9, max_iterations=20):
        from scipy.stats import norm
        if S <= 1e-9 or F <= 1e-9: return None
        if t > 1e-9:
            rd_minus_rf = np.log(F / S) / t
            rf = rd - rd_minus_rf  # Calculate rf
        else:
            rd_minus_rf = 0.0
            rf = rd

        def f(x):
            d1 = (np.log(S / x) + (rd_minus_rf + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
            # SPOT DELTA for Call: exp(-rf*t) * N(d1)
            return np.exp(-rf * t) * N(d1) - delta
            
        def df(x):
            d1 = (np.log(S / x) + (rd_minus_rf + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
            # Derivative of spot delta w.r.t. K
            return -np.exp(-rf * t) * norm.pdf(d1) / (x * sigma * np.sqrt(t))
        
        x = guess
        for i in range(max_iterations):
            # print(f'Iteration no for call: {i}, Current guess: {x}')
            fx = f(x)
            fpx = df(x)
            if abs(fpx) < 1e-12: break
            x_next = x - fx / fpx
            if abs(x_next - x) < tolerance:
                return x_next
            x = x_next
        return None



    def Imply_K_put_F(self, guess, S, F, rd, sigma, t, delta, tolerance=1e-9, max_iterations=20):
        from scipy.stats import norm
        if S <= 1e-9 or F <= 1e-9: return None
        if t > 1e-9:
            rd_minus_rf = np.log(F / S) / t
            rf = rd - rd_minus_rf  # Calculate rf
        else:
            rd_minus_rf = 0.0
            rf = rd

        def f(x):
            d1 = (np.log(S / x) + (rd_minus_rf + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
            # SPOT DELTA for Put: -exp(-rf*t) * N(-d1)
            # Note: Put delta is negative, so we're solving: -exp(-rf*t)*N(-d1) - delta = 0
            return -np.exp(-rf * t) * N(-d1) - delta
            
        def df(x):
            d1 = (np.log(S / x) + (rd_minus_rf + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
            # Derivative of put spot delta w.r.t. K
            return -np.exp(-rf * t) * norm.pdf(d1) / (x * sigma * np.sqrt(t))
        
        x = guess
        for i in range(max_iterations):
            # print(f'Iteration no for put: {i}, Current guess: {x}')
            fx = f(x)
            fpx = df(x)
            if abs(fpx) < 1e-12: break
            x_next = x - fx / fpx
            if abs(x_next - x) < tolerance:
                return x_next
            x = x_next
        return None

    # --- MODIFIED: guess_delta2_F (Uses Forward Price F) ---
    def guess_delta2_F (self, S, K, F, rd, vol_curve, delta_adjust, expire, delivery): # 找出不是bucket tenor的implied vol
        """
        Iteratively finds the volatility that makes the implied strike K
        match the target strike K. Uses Forward Price (F) instead of rf.
        """
        from scipy import interpolate
        dayCount = 365 if self.CCY in self.dayCount365 else 360
        t = expire/dayCount
        
        # Calculate implied rd - rf term
        if S <= 1e-9 or F <= 1e-9: return -1.0, 0.0
        if t > 1e-9:
            rd_minus_rf = np.log(F / S) / t
        else:
            rd_minus_rf = 0.0
            
        guessDelta = -0.25 if S < K else -0.75
        
        while True:
            guessDelta_imply_vol = interpolate.interp1d(delta_adjust, vol_curve, kind='quadratic', fill_value="extrapolate")(guessDelta)
            
            # Use the NEW Imply_K_put_F function
            imply_k = self.Imply_K_put_F(S, S, F, rd, guessDelta_imply_vol, t, guessDelta, tolerance=1e-5, max_iterations=100)
            
            if imply_k is None:
                if S < K: return -0.999, vol_curve[0]
                elif S > K: return -0.001, vol_curve[4]
            
            imply_k = (imply_k + K)/2
            
            # Recalculate d1 and guessDelta using implied rd - rf
            d1 = (np.log(S / imply_k) + (rd_minus_rf + 0.5 * guessDelta_imply_vol ** 2) * t) / (guessDelta_imply_vol * np.sqrt(t))
            
            # maybe corrected ver
            d1 = (np.log(S / imply_k) + (rd_minus_rf + 0.5 * guessDelta_imply_vol ** 2) * t) / (guessDelta_imply_vol * np.sqrt(t))

            # Recalculate Delta (Put K-Delta) for the new implied_k
            guessDelta = -imply_k / S * np.exp(-rd * t) * N(-d1 + guessDelta_imply_vol * np.sqrt(t))
            # maybe corrected ver
            rf = rd - rd_minus_rf
            guessDelta = -np.exp(-rf * t) * N(-d1) 
            residual = imply_k - K
            
            if abs(residual) < 10**-5:
                # print(guessDelta, guessDelta_imply_vol)
                break
                
        return guessDelta, guessDelta_imply_vol

    # --- MODIFIED: imply_vol3_F (Uses Market Forward F) --- # 此版本為input固定匯率，求出implied vol
    def imply_vol3_F(self, currency, Trade_date, fixing_date, value_date, S, K, market_forward_price):
        def date_diff(date_list):
            return [i.days for i in [i - date_list[0] for i in date_list]]

        from scipy import interpolate
        revCcy = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
        dayCount = 365 if currency in ['GBPUSD', 'AUDNZD', 'NZDUSD', 'USDZAR', 'USDCAD'] else 360
        # S = spot[currency][Trade_date]
        s_date = self.spotDate.loc[Trade_date][currency]
        # change s_date back to datetime
        s_date = pd.to_datetime(s_date)
        

        vol_exercise_date = self.volExpDate.loc[currency]
        value_to_spot_days = (value_date - s_date).days
        # vol_exercise_date.loc[Trade_date][0] 為今日日期, fixing date是option fixing date
        fixing_to_trade_days = (fixing_date - vol_exercise_date.loc[Trade_date][0]).days
        if fixing_to_trade_days == 0:
            fixing_to_trade_days = 1 # avoid zero division
        fixing_to_trade_days_year = fixing_to_trade_days/dayCount

        fixToTradeDate = (fixing_date - Trade_date).days
        settleToTradeDate = (value_date - s_date).days
        # 1. Volatility Curve Interpolation (No change needed here)

        vol = self.vol.loc[currency]
        b = vol_exercise_date.loc[Trade_date]
        b = date_diff(b)
        vol_at_fixing_date = []
        # print(vol)
        for i in vol.columns: # 用多項式插值法找出price date至fixing date各market tenor delta的implied vol
            c = vol.loc[Trade_date, i]
            # print(b, c)
            # b: 當日的option到期日, c: 當日option該天期的market implied vol
            # spline 差補
            # fun_vol = interpolate.interp1d(b, c, kind='quadratic', fill_value="extrapolate")

            # linear 差補
            fun_vol = interpolate.interp1d(b, c, kind='linear', fill_value="extrapolate")
            vol_at_fixing_date.append(fun_vol(fixing_to_trade_days) * 1)
        vol_at_fixing_date = [i/100 for i in vol_at_fixing_date]

        # 2. Domestic Rate (rd) Interpolation (No change needed here)
        # 以下會找出pricing date美元插補利率
        local_currency = currency
        RD = self.usdRate.loc["USD"].loc[Trade_date] # usd is always domestic rate?

        RD_tenor = [(i-s_date).days for i in self.rateDate.loc[local_currency].loc[Trade_date]]
        # print(RD)
        # print(RD_tenor)
        RD_tenor = pd.Series(RD_tenor)[[~np.isnan(i) for i in RD]]
        RD = RD[[~np.isnan(i) for i in RD]]
        fun_RD = interpolate.interp1d(RD_tenor, RD, kind='quadratic', fill_value="extrapolate")
        rd = fun_RD(value_to_spot_days)/100 # 選擇權value date與今日spot date的天數差異是插補目標

        # 3. Market Forward Price (F)
        F = market_forward_price # <--- **This is the key new input**
        if S <= 1e-9 or F <= 1e-9: return 0.0
        if fixing_to_trade_days_year > 1e-9:
            if currency[:3] == 'USD':
                rf_implied = (1.0 / fixing_to_trade_days_year) * ((F / S) * (1.0 + rd * fixing_to_trade_days_year) - 1.0)
            if currency[3:] == 'USD':
                rf_implied = (1.0 / fixing_to_trade_days_year) * ((S / F) * (1.0 + rd * fixing_to_trade_days_year) - 1.0)
            # rd_minus_rf = np.log(F / S) / fixing_to_trade_days_year #從swap及美元利率得到另一端利率
            rd_minus_rf = rd - rf_implied
        else:
            rf_implied = rd 
            rd_minus_rf = rd - rf_implied


        ## 帶入參數區 (須將利率都轉成continuous compounding)
        # rd_cont = np.log(1 + rd * fixing_to_trade_days_year) / fixing_to_trade_days_year
        # rf_implied_cont = np.log(1 + rf_implied * fixing_to_trade_days_year) / fixing_to_trade_days_year

        # 20251125更新
        rd_cont = (365/value_to_spot_days) * np.log(1 + rd * (value_to_spot_days/365))
        rf_implied_cont = (365/value_to_spot_days) * np.log(1 + rf_implied * (value_to_spot_days/365))

        rd_minus_rf_cont = rd_cont - rf_implied_cont
        # 5. Calculate Standard Delta Strikes using the NEW Imply_K_call_F and implied rf
        # Use the implied rf for K50 calculation (where F = S * exp((rd - rf)*t) )
        
        # 10D call
        K90 = self.Imply_K_call_F(guess=S, S=S, F=F, rd=rd_cont, sigma=vol_at_fixing_date[0], t=fixing_to_trade_days_year, delta=0.1)
        
        # 25D call
        K75 = self.Imply_K_call_F(guess=S, S=S, F=F, rd=rd_cont, sigma=vol_at_fixing_date[1], t=fixing_to_trade_days_year, delta=0.25)
        
        # K50 (ATM) calculation using implied rf
        K50 = S * np.exp((rd_minus_rf_cont - 0.5 * vol_at_fixing_date[2] ** 2) * fixing_to_trade_days_year)

        # print(f'10D call: {K90}, 25D call: {K75}, ATM: {K50}')
        # 6. Delta Adjust Calculation (No change needed as it's based on the resulting K)
        # 因call及put在不同曲線下，所以把delta統一都轉為put delta進行插補
        delta_adjust = [0.1 - K90 / S * np.exp(-rd_cont * fixing_to_trade_days_year),
                        0.25 - K75 / S * np.exp(-rd_cont * fixing_to_trade_days_year),
                        -0.5 * K50 / S * np.exp(-rd_cont * fixing_to_trade_days_year),
                        -0.25,
                        -0.1]

        # print(delta_adjust)
        # 7. Implied Volatility Search (Use the NEW guess_delta2_F)
        # The K_target (your 65D strike) is passed as K to this function.
        K_imply_delta, K_imply_vol = self.guess_delta2_F(S, K, F, rd_cont, vol_at_fixing_date, delta_adjust, fixing_to_trade_days, value_to_spot_days)
        K_imply_delta_call = 1 + K_imply_delta if K_imply_delta < 0 else 1 - K_imply_delta

        K_fromDelta = self.Imply_K_call_F(
            guess=K, 
            S=S, 
            F=F, 
            rd=rd_cont, 
            sigma=K_imply_vol, # Use the final converged vol
            t=fixing_to_trade_days_year, 
            delta=K_imply_delta_call # The required target Delta
        )
        
        # print the element in results to check where is the problem
        # print(f'sigmaImplied: {K_imply_vol}, K: {K}, rd: {rd}, rf_implied: {rf_implied}, fixing_to_trade_days_year: {fixing_to_trade_days_year}, S: {S}, F: {F}, vol_at_fixing_date: {vol_at_fixing_date}, K_imply_delta: {K_imply_delta}')

        
        # [3] Return the results as a dictionary
        results = {
            'ccy': currency,
            'sigmaImplied': K_imply_vol,
            'K': K,
            'rf_implied': rf_implied,  # The implied foreign rate calculated earlier
            'rd': rd,                  # The interpolated domestic rate
            'T_years': fixing_to_trade_days_year,
            'S': S,
            'F': F,
            'impliedVol': vol_at_fixing_date,
            'impliedDelta': K_imply_delta,
            'priceToFixDate': fixToTradeDate,
            'priceToSettleDate': settleToTradeDate
        }

        return results
    



    def getOptionParams(self, K, strgy, priceDate, expireDate):
        PRICE_DATE = priceDate
        EXP_DATE = expireDate
        CCY = self.CCY
        DELIV_DATE = self.spotDate.loc[EXP_DATE][CCY]
        DELIV_DATE = pd.to_datetime(DELIV_DATE).strftime('%Y-%m-%d')

        
        if self.ccyLeft.upper() == "USD":
            foreday_basis = 360
            cptyday_basis = 365 if self.dayCount365 else 360
        if self.ccyRight.upper() == "USD":
            cptyday_basis = 360
            foreday_basis = 365 if self.dayCount365 else 360
            


        # Example Data (MUST include Spot, T=0)
        spotData = self.spot.loc[PRICE_DATE, CCY]
        dayTenorData = self.fwd.loc[CCY].loc[PRICE_DATE]
        dayTenorData = pd.concat([pd.Series([spotData], index=['SP']), dayTenorData])

        # concat 
        tdyTenor = self.fwdDate.loc[CCY].loc[PRICE_DATE].dropna()
        # get the diff between value date and spot date
        spTmp = tdyTenor.name
        tenorDayTmp = tdyTenor.values

        value_to_spot_days = []
        for i in tenorDayTmp:
            delta_days = (i - spTmp).days
            value_to_spot_days.append(delta_days)

        # tenorAcutalDay = [0, 1, 7, 14, 21, 30, 60, 91, 121, 182, 273, 365]
        tenorAcutalDay = value_to_spot_days

        tenors_days = tenorAcutalDay
        forward_prices = dayTenorData.values

        # Create the interpolation function
        forward_curve = self.create_forward_curve(tenors_days, forward_prices)
        spot = self.spot.loc[PRICE_DATE, CCY]

        d1 = datetime.strptime(PRICE_DATE, '%Y-%m-%d')
        d2 = datetime.strptime(DELIV_DATE,   '%Y-%m-%d')

        delta_days = (d2 - d1).days

        T_target = delta_days
        T_target = T_target / 360 # ~ 0.493 years
        F_interpolated = forward_curve(T_target)
        # fwd = FORWARD.loc[CCY].loc[PRICE_DATE, '1M']
        fwd = F_interpolated
        K = K


        results = self.imply_vol3_F(
            currency=CCY, 
            Trade_date=pd.to_datetime(PRICE_DATE), 
            fixing_date=pd.to_datetime(EXP_DATE), # 29 days from 2012-09-03 (example)
            value_date=pd.to_datetime(DELIV_DATE), # Delivery date (e.g., T+2 from fixing)
            S=spot,
            K=K, # The strike that corresponds to the 65D on the smile
            market_forward_price=fwd
        )

        # append additional info (date)
        results['Trade_date'] = PRICE_DATE
        results['fixing_date'] = EXP_DATE
        results['value_date'] = DELIV_DATE

        # get greeks
        bsRes = self.getOptionGreeks(
        s=results['S'],
        K=K,
        F=fwd,
        sigma=results['sigmaImplied'],
        expire=results['priceToFixDate'],
        delivery=results['priceToSettleDate'],
        usd_rate=results['rd'],
        foreday_basis=foreday_basis,
        cptyday_basis=cptyday_basis,
        strgy=strgy,
        ccy_left=self.ccyLeft,
        ccy_right=self.ccyRight,
        date=None
        )

        self.optionParams = results
        self.optionGreeks = bsRes




    def getOptionGreeks(
        self, s,K,F,sigma,expire,delivery,usd_rate,foreday_basis,cptyday_basis,strgy,ccy_left,ccy_right,date=None,
    ):

        import math
        import numpy as np
        from scipy.stats import norm

        N_pdf = stats.norm.pdf
        N = stats.norm.cdf
        # Basic sanity checks
        if expire <= 0:
            raise ValueError("expire must be > 0")
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        if s <= 0 or K <= 0:
            raise ValueError("s and K must be > 0")
        if foreday_basis <= 0 or cptyday_basis <= 0:
            raise ValueError("day count bases must be > 0")

        base = 365.0 if self.CCY in self.dayCount365 else 360.0
        # base = 360.0
        t = expire / base  # time in years
        T_annual = expire / base  # annual fraction used for implied rate calc


        # change usd_rate from simple to continuous rate first
        usd_rate = np.log(1 + usd_rate * (expire / base)) / (expire / base)
        
        # Determine which side is USD and compute implied counterparty rate
        if str(ccy_left).upper() == "USD":
            implied_ccy_rate = usd_rate + np.log(F / s) / T_annual
            fore_rate = usd_rate
            cpty_rate = implied_ccy_rate
        elif str(ccy_right).upper() == "USD":
            implied_ccy_rate = usd_rate - np.log(F / s) / T_annual
            fore_rate = implied_ccy_rate
            cpty_rate = usd_rate
        else:
            raise ValueError("Function expects USD to be either ccy_left or ccy_right")

        # print(fore_rate, cpty_rate)
        # Compute yields (following the original logic)
        if delivery > cptyday_basis:
            CtrYield = base / expire * (delivery / cptyday_basis) * math.log(1 + cpty_rate)
        else:
            CtrYield = base / expire * math.log(1 + cpty_rate * delivery / cptyday_basis)

        if delivery > foreday_basis:
            Yield = base / expire * (delivery / foreday_basis) * math.log(1 + fore_rate)
        else:
            Yield = base / expire * math.log(1 + fore_rate * delivery / foreday_basis)

        # print(Yield, CtrYield)
        # Handle zero volatility or zero time to expiry
        if sigma == 0 or t == 0:
            # Intrinsic-like price with discounting using yields from above
            if strgy.lower() == "call":
                return max(0.0, s * math.exp(-Yield * t) - K * math.exp(-CtrYield * t))
            elif strgy.lower() == "put":
                return max(0.0, K * math.exp(-CtrYield * t) - s * math.exp(-Yield * t))
            else:
                raise ValueError("strgy must be 'Call' or 'Put'")

        sqrt_t = math.sqrt(t)
        d1 = (math.log(s / K) + ((CtrYield - Yield) + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        bsDict = dict.fromkeys(['pxPips', 'px%', 'delta', 'gamma', 'vega', 'theta', 'rho_rd', 'rho_rf'], 0.0)

        if strgy.lower() == "call":
            delta = math.exp(-Yield * t) * norm.cdf(d1)
            european_option = s * delta - K * math.exp(-CtrYield * t) * norm.cdf(d2)
            # return european_option
        elif strgy.lower() == "put":
            delta = -math.exp(-Yield * t) * norm.cdf(-d1)
            european_option = K * math.exp(-CtrYield * t) * norm.cdf(-d2) + s * delta
            # return european_option
        else:
            raise ValueError("strgy must be 'Call' or 'Put'")

        I = 1 if strgy.lower() == "call" else -1
        term1 = -s * N_pdf(d1) * sigma * np.exp(-Yield * t) / (2.0 * sqrt_t)
        term2 = I * Yield * s * N(I * d1) * np.exp(-Yield * t)
        term3 = -I * CtrYield * K * np.exp(-CtrYield * t) * N(I * d2)

        bsDict['ccy'] = self.CCY
        bsDict['spot'] = s
        bsDict['pxPips'] = european_option
        bsDict['px%'] = european_option / s * 100
        bsDict['cp'] = strgy.lower()
        bsDict['delta'] = delta
        bsDict['gamma'] = N_pdf(d1) * np.exp(-Yield * t) / (s * sigma * sqrt_t)
        bsDict['vega'] = s * np.exp(-Yield * t) * sqrt_t * N_pdf(d1)
        bsDict['theta'] = term1 + term2 + term3
        bsDict['rho_rd'] = I * K * t * np.exp(-CtrYield * t) * N(I * d2)
        bsDict['rho_rf'] = -I * s * t * np.exp(-Yield * t) * N(I * d1)

        return bsDict

