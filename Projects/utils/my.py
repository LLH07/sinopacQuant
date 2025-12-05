import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# get trading data from daily portChange
def getStats(totalCount, dailyS):
    s = pd.Series()

    if totalCount >= 252:
        s['Mean'] = (dailyS[0:].sum()) / totalCount * 252
        s['Std'] = (dailyS[0:].std()) * np.sqrt(252)
    else:
        s['Mean'] = (dailyS[0:].sum()) / totalCount * totalCount
        s['Std'] = (dailyS[0:].std()) * np.sqrt(totalCount)

    s['Sharpe'] = s['Mean'] / s['Std']
                                                                                                 
    return s


def getMdd(daily_returns):
    cumulative_returns = daily_returns.cumsum()
    running_max = cumulative_returns.cummax()
    drawdown = running_max - cumulative_returns
    mdd_end_idx = drawdown.idxmax()
    mdd_value = drawdown[mdd_end_idx]
    mdd_start_idx = running_max.loc[:mdd_end_idx].idxmax()
    mdd_duration = (mdd_end_idx - mdd_start_idx).days
    return {
        "mdd_value": mdd_value,
        "mdd_start_date": mdd_start_idx,
        "mdd_end_date": mdd_end_idx,
        "mdd_duration": mdd_duration
    }

def getCMTKSpot(ccys, startDate, endDate):
    from xbbg import blp
    dr = pd.date_range(start=startDate, end=endDate, freq='B')
    spotDf = pd.DataFrame(index=dr, columns=ccys)
    # print(spotDf.shape)
    for ccy in ccys:
        spot = blp.bdh(tickers=f'{ccy} CMTK Curncy', flds=["PX_LAST"], start_date=startDate, end_date=endDate)
        spot = spot.reindex(dr).ffill()
        spotDf[ccy] = spot
    return spotDf

def getBFixTokyoSpot(ccys, startDate, endDate):
    from xbbg import blp
    dr = pd.date_range(start=startDate, end=endDate, freq='B')
    spotDf = pd.DataFrame(index=dr, columns=ccys)
    # print(spotDf.shape)
    for ccy in ccys:
        spot = blp.bdh(tickers=f'{ccy} T150 Curncy', flds=["PX_LAST"], start_date=startDate, end_date=endDate)
        spot = spot.reindex(dr).ffill()
        spotDf[ccy] = spot
        
    return spotDf

def getOHLC_BGN(ccys, startDate, endDate):
    from xbbg import blp
    dr = pd.date_range(start=startDate, end=endDate, freq='B')
    tmpDf = pd.DataFrame(index=dr, columns=ccys)
    ohlcD = dict.fromkeys(ccys, tmpDf)
    for ccy in ccys:
        ohlc = blp.bdh(tickers=f'{ccy} BGN Curncy', flds=["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"], start_date=startDate, end_date=endDate)
        ohlc = ohlc.reindex(dr).ffill()
        # rename the columns to 'open', 'high', 'low', 'close'
        ohlc.columns = ['open', 'high', 'low', 'close']
        ohlcD[ccy] = ohlc

    return ohlcD

def getStrgyDailyChange(signalDf, btDailyRet, startDate, endDate, slipeRate=0.0005):
    import datetime
    dr = pd.date_range(start=startDate, end=endDate, freq='B')
    
    # Fix: Reindex instead of resample with DatetimeIndex
    tmp = signalDf.reindex(dr).ffill()
    
    portDailyChange = btDailyRet * tmp.shift(1)
    portDailyChange = portDailyChange * (1-slipeRate)
    portDailyChange['port'] = portDailyChange.sum(axis=1)
    portDailyChange['port'].cumsum().plot(label='Cumulative Return', color='black', figsize=(20, 5))
    plt.title('Portfolio Cumulative Return')
    plt.show()
    
    return portDailyChange


def getDetailedTradingStats(portDailyChange, startDate, endDate):
    portDailyChange = portDailyChange[(portDailyChange.index >= startDate) & (portDailyChange.index <= endDate)]
    yearlyRes = pd.DataFrame()

    yearlyRes['return(%)'] = round(portDailyChange.resample('YE').sum().reset_index(drop=True) * 100, 2)
    yearlyRes['dailyStd(%)'] = round(portDailyChange.resample('YE').std().reset_index(drop=True)* 100, 2)
    yearlyRes['yearlyMDD(%)'] = round(portDailyChange.resample('YE').apply(lambda x: getMdd(x)['mdd_value']).reset_index(drop=True)* 100, 2)
    yearlyRes['yearlySharpe'] = round(portDailyChange.resample('YE').apply(lambda x: getStats(x.shape[0], x)['Sharpe']).reset_index(drop=True), 3)
    yearlyRes['yearlyProfit(pos=USD1mio)'] = yearlyRes['return(%)'].astype(float).apply(lambda x: '{:,.0f}'.format(x*10000))
    yearlyRes['dailyStd(pos=USD1mio)'] = yearlyRes['dailyStd(%)'].astype(float).apply(lambda x: '{:,.0f}'.format(x*10000))


    yearlyRes.index = portDailyChange.index.year.unique()


    yearlyRes.loc['回測期間', 'return(%)'] = round(getStats(totalCount=portDailyChange.dropna().shape[0], dailyS=portDailyChange)['Mean']*100, 2)
    yearlyRes.loc['回測期間', 'dailyStd(%)'] = round(portDailyChange.std()* 100, 2)
    yearlyRes.loc['回測期間', 'yearlyMDD(%)'] = round(getMdd(portDailyChange)['mdd_value'] * 100, 2)
    yearlyRes.loc['回測期間', 'yearlySharpe'] = round(getStats(totalCount=portDailyChange.dropna().shape[0], dailyS=portDailyChange)['Sharpe'], 3)
    yearlyRes.loc['回測期間', 'yearlyProfit(pos=USD1mio)'] = yearlyRes.loc['回測期間', 'return(%)'].astype(float)/100*1000000
    yearlyRes.loc['回測期間', 'yearlyProfit(pos=USD1mio)'] = '{:,.0f}'.format(yearlyRes.loc['回測期間', 'yearlyProfit(pos=USD1mio)'])

    yearlyRes.loc['回測期間', 'dailyStd(pos=USD1mio)'] = yearlyRes.loc['回測期間', 'dailyStd(%)'].astype(float)/100*1000000
    yearlyRes.loc['回測期間', 'dailyStd(pos=USD1mio)'] = '{:,.0f}'.format(yearlyRes.loc['回測期間', 'dailyStd(pos=USD1mio)'])

    return yearlyRes



# def plotBtRes(portDailyChange, startDate, endDate):
#     # import chinese
#     import matplotlib
#     matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

#     # show minus sign
#     matplotlib.rcParams['axes.unicode_minus'] = False

#     stats = getDetailedTradingStats(portDailyChange, startDate, endDate)
#     def getMdd(daily_returns):
#         cumulative_returns = daily_returns.cumsum()
#         running_max = cumulative_returns.cummax()
#         drawdown = running_max - cumulative_returns
#         mdd_end_idx = drawdown.idxmax()
#         mdd_value = drawdown[mdd_end_idx]
#         mdd_start_idx = running_max.loc[:mdd_end_idx].idxmax()
#         mdd_duration = (mdd_end_idx - mdd_start_idx).days
#         return {
#             "mdd_value": mdd_value,
#             "mdd_start_date": mdd_start_idx,
#             "mdd_end_date": mdd_end_idx,
#             "mdd_duration": mdd_duration
#         }, cumulative_returns, running_max, drawdown
    
#     # Get MDD and related data
#     mdd_info, cumulative_returns, running_max, drawdown = getMdd(portDailyChange)

#     # Plot cumulative returns and drawdown
#     plt.figure(figsize=(12, 6))
#     plt.plot(cumulative_returns, label="Cumulative Returns", color="blue")
#     plt.plot(running_max, label="Running Max", color="green", linestyle="--")

#     # Highlight the MDD period
#     plt.fill_between(cumulative_returns.index, cumulative_returns, running_max, 
#                     where=(cumulative_returns.index >= mdd_info["mdd_start_date"]) & 
#                         (cumulative_returns.index <= mdd_info["mdd_end_date"]),
#                     color="red", alpha=0.3, label="Drawdown")

#     # Annotate MDD details
#     plt.annotate(
#         f"MDD: {mdd_info['mdd_value']*100:.2f}%\nStart: {mdd_info['mdd_start_date']}\nEnd: {mdd_info['mdd_end_date']}",
#         xy=(mdd_info["mdd_end_date"], cumulative_returns[mdd_info["mdd_end_date"]]),
#         xytext=(-50, -50),
#         textcoords="offset points",
#         arrowprops=dict(arrowstyle="->", color="red"),
#         fontsize=10,
#         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
#     )

#     # Annotate the Sharpe ratio, MDD, and yearly return
#     yearly_return = stats.loc['回測期間', 'return(%)']
#     mdd = stats.loc['回測期間', 'yearlyMDD(%)']
#     sharpe_ratio = stats.loc['回測期間', 'yearlySharpe']

#     # Annotate the stats on the plot
#     annotation_text = (
#         f"Yearly Return: {yearly_return:.2f}%\n"
#         f"MDD: {mdd:.2f}%\n"
#         f"Sharpe Ratio: {sharpe_ratio:.2f}"
#     )

#     plt.annotate(
#         annotation_text,
#         xy=(0.05, 0.95),  # Position of annotation (relative to plot)
#         xycoords='axes fraction',
#         fontsize=15,
#         ha='left',  # Horizontal alignment
#         va='top',   # Vertical alignment
#         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="yellow"), 
#     )

#     # Add labels, grid, legend, and title
#     plt.title("策略回測結果")
#     plt.xlabel("Date")
#     plt.ylabel("Cumulative Returns")
#     plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
#     plt.grid(True)  # Add grid to the plot
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


import matplotlib.pyplot as plt
import pandas as pd
from xbbg import blp


def plotBtRes(portDailyChange, startDate, endDate):
    # Import Chinese font settings
    from datetime import datetime
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # Show minus sign

    portDailyChange = portDailyChange[(portDailyChange.index >= startDate) & (portDailyChange.index <= endDate)]
    stats = getDetailedTradingStats(portDailyChange, startDate, endDate)
    
    def getMdd(daily_returns):
        cumulative_returns = daily_returns.cumsum()
        running_max = cumulative_returns.cummax()
        drawdown = running_max - cumulative_returns
        mdd_end_idx = drawdown.idxmax()
        mdd_value = drawdown[mdd_end_idx]
        mdd_start_idx = running_max.loc[:mdd_end_idx].idxmax()
        mdd_duration = (mdd_end_idx - mdd_start_idx).days
        return {
            "mdd_value": mdd_value,
            "mdd_start_date": mdd_start_idx,
            "mdd_end_date": mdd_end_idx,
            "mdd_duration": mdd_duration
        }, cumulative_returns, running_max, drawdown

    # Get MDD and related data
    mdd_info, cumulative_returns, running_max, drawdown = getMdd(portDailyChange)

    # Fetch DXY data using Bloomberg API
    dxy_data = blp.bdh('DXY Index', 'PX_LAST', startDate, endDate)
    dxy_data.columns = dxy_data.columns.droplevel(0)  # Drop "DXY Index" level, keeping "DXY"
    dxy_data.index = pd.to_datetime(dxy_data.index)

    # Create the figure and primary axis (left y-axis)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot cumulative returns and drawdown on the left y-axis
    ax1.plot(cumulative_returns, label="Cumulative Returns", color="blue")
    ax1.plot(running_max, label="Running Max", color="green", linestyle="--")
    ax1.fill_between(
        cumulative_returns.index,
        cumulative_returns,
        running_max,
        where=(cumulative_returns.index >= mdd_info["mdd_start_date"]) & 
              (cumulative_returns.index <= mdd_info["mdd_end_date"]),
        color="red",
        alpha=0.3,
        label="Drawdown"
    )
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Cumulative Returns", color="blue", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
    #ax1.legend(loc="upper left")

    # Create a secondary axis (right y-axis) for DXY
    ax2 = ax1.twinx()
    ax2.plot(dxy_data['PX_LAST'], label="DXY Index", color="purple", alpha=0.6)
    ax2.set_ylabel("DXY Index", color="purple", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="purple")
    #ax2.legend(loc="upper right")

    # Annotate MDD details
    ax1.annotate(
        f"MDD: {mdd_info['mdd_value']*100:.2f}%\nStart: {datetime.strftime(mdd_info['mdd_start_date'], '%Y-%m-%d')}\nEnd: {datetime.strftime(mdd_info['mdd_end_date'], '%Y-%m-%d')}",
        xy=(mdd_info["mdd_end_date"], cumulative_returns[mdd_info["mdd_end_date"]]),
        xytext=(-50, -50),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    )

    # annotate out of sample

    # Annotate the Sharpe ratio, MDD, and yearly return
    yearly_return = stats.loc['回測期間', 'return(%)']
    mdd = stats.loc['回測期間', 'yearlyMDD(%)']
    sharpe_ratio = stats.loc['回測期間', 'yearlySharpe']

    annotation_text = (
        f"Yearly Return: {yearly_return:.2f}%\n"
        f"MDD: {mdd:.2f}%\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}"
    )

    ax1.annotate(
        annotation_text,
        xy=(0.05, 0.95),  # Position of annotation (relative to plot)
        xycoords='axes fraction',
        fontsize=15,
        ha='left',  # Horizontal alignment
        va='top',   # Vertical alignment
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="yellow")
    )

    # Add title and grid
    plt.title("策略回測結果", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plotStrgyYearlyReturn(portDailyChange, startYear, endYear, dxyCompare=True):
    from xbbg import blp
    import matplotlib.pyplot as plt
    import pandas as pd
    # 導入中文
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

    # show minus sign
    matplotlib.rcParams['axes.unicode_minus'] = False

    startDate = f'{startYear}-01-01'
    endDate = f'{endYear}-12-31'

    dr = pd.date_range(start=startDate, end=endDate, freq='B') 

    if dxyCompare:
        dxy = blp.bdh('DXY Index', 'PX_LAST', startDate, endDate)
        dxy.rename(columns={'DXY Index':'dxy'}, inplace=True)
        dxy.index = pd.DatetimeIndex(dxy.index)

    years = range(int(startYear), int(endYear) + 1)
    n_years = len(years)

    # Calculate subplot grid dimensions
    # For nice layout, determine rows and columns
    import math
    cols = min(2, n_years)  # Maximum 2 columns
    rows = math.ceil(n_years / cols)

    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), sharex=False)

    # Flatten axes array for easier iteration if multiple rows and columns
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1 and cols > 1:
        axes = axes  # Already a 1D array
    elif rows > 1 and cols == 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Convert single axis to list

    # Plot each year in its own subplot
    for i, year in enumerate(years):
        if i < len(axes):  # Ensure we don't exceed available subplots
            ax1 = axes[i]
            ax2 = ax1.twinx()
            
            # Get data for the year
            year_data = portDailyChange[portDailyChange.index.year == year]
            spot_data = dxy[dxy.index.year == year]
            
            # Plot equity curve on left y-axis
            equity_line, = ax1.plot(year_data.index, year_data.cumsum(), 
                    label='Equity Curve', color='black')
            
            # Plot USDJPY spot on right y-axis
            spot_line, = ax2.plot(spot_data.index, spot_data[('dxy', 'PX_LAST')], 
                    label='DXY', color='grey')
            
            # Set labels
            ax1.set_xlabel('Date', fontsize=14)
            ax1.set_ylabel('Cumulative Portfolio Change', fontsize=14, color='black')
            ax2.set_ylabel('DXY', fontsize=14, color='grey')
            
            # Add title for each subplot
            ax1.set_title(f'整體投組策略表現_{year}', fontsize=16, color='black')
            
            # Create legends
            lines = [equity_line, spot_line]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='upper left')

            # set a text to show the corr between portDailyChange and USDJPY
            corr = year_data.rolling(60).corr(spot_data[('dxy', 'PX_LAST')]).dropna().mean()
            ax1.text(0.5, 0.9, f'Rolling 60D Correlation: {corr:.2f}', ha='center', va='center', transform=ax1.transAxes, fontsize=15, color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='red'))

    # Remove any empty subplots
    for i in range(len(years), len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout
    plt.tight_layout()
    plt.show()



# other market data
def getRSI(ccys, startDate, endDate):
    import pandas as pd

    basketSpot = getCMTKSpot(ccys, startDate, endDate)

    def calculate_rsi_wilder(prices, period=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_currency_rsi(df, period=14):
        rsi_df = pd.DataFrame(index=df.index)
        for currency in df.columns:
            rsi_df[currency + '_RSI'] = calculate_rsi_wilder(df[currency], period)
        return rsi_df

    rsi_results = calculate_currency_rsi(basketSpot)
    return rsi_results