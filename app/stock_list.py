"""
stock_list.py – Curated list of NSE stock symbols used by the AITrader scanner.

Symbols are in Yahoo Finance format (suffix ".NS").  All internal logic and
the database use the BASE symbol (see app/symbols.py); conversion happens at
the yfinance / Shoonya edges.

The universe is ~145 names (NIFTY 50 + Next 50 selection + mid-caps); the
1,500-stock expansion is future work.
"""

NSE_SYMBOLS: list[str] = [
    # NIFTY 50
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
    "BAJFINANCE.NS", "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "TECHM.NS",
    # TATAMOTORS became TMPV (Tata Motors Passenger Vehicles) after the
    # Oct-2025 CV demerger; the commercial-vehicle entity lists as TMLCV.
    "TMPV.NS", "TATASTEEL.NS", "ADANIPORTS.NS", "JSWSTEEL.NS", "COALINDIA.NS",
    "M&M.NS", "BAJAJFINSV.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "BPCL.NS", "GRASIM.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "BAJAJ-AUTO.NS",
    "BRITANNIA.NS", "HINDALCO.NS", "UPL.NS", "SHREECEM.NS", "VEDL.NS",

    # NIFTY NEXT 50 (selection)
    # ADANITRANS was renamed to ADANIENSOL (Adani Energy Solutions) in 2023.
    "ADANIENT.NS", "ADANIGREEN.NS", "ADANIENSOL.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "BANDHANBNK.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS", "CHOLAFIN.NS",
    "COLPAL.NS", "CONCOR.NS", "DABUR.NS", "DLF.NS", "GAIL.NS",
    "GODREJCP.NS", "HAVELLS.NS", "ICICIGI.NS", "ICICIPRULI.NS", "INDUSTOWER.NS",
    "INDIGO.NS", "IOC.NS", "IRCTC.NS", "JUBLFOOD.NS", "LICHSGFIN.NS",
    # MCDOWELL-N was renamed to UNITDSPR (United Spirits), already listed below.
    "LUPIN.NS", "MARICO.NS", "MUTHOOTFIN.NS", "NAUKRI.NS",
    "NMDC.NS", "PAGEIND.NS", "PETRONET.NS", "PIDILITIND.NS", "PNB.NS",
    "RECLTD.NS", "SIEMENS.NS", "SRF.NS", "TATACOMM.NS", "TORNTPHARM.NS",
    # ZOMATO was renamed ETERNAL in 2025.
    "TRENT.NS", "UNITDSPR.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "ETERNAL.NS",

    # Mid-cap selection
    "ABCAPITAL.NS", "ALKEM.NS", "ATUL.NS", "BALKRISIND.NS", "BATAINDIA.NS",
    "CANBK.NS", "DEEPAKNTR.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS",
    # GMRINFRA was renamed GMRAIRPORT (GMR Airports) in 2024.
    "FORTIS.NS", "GLAXO.NS", "GMRAIRPORT.NS", "GRANULES.NS", "HAL.NS",
    "HINDPETRO.NS", "HONAUT.NS", "IDFCFIRSTB.NS", "IGL.NS", "INDIANB.NS",
    # JUBILANT became JUBLPHARMA (Jubilant Pharmova, 2021); MINDTREE merged
    # into LTIMindtree (LTIM, 2022); MOTHERSUMI became MOTHERSON (Samvardhana
    # Motherson International, 2022).
    "INDHOTEL.NS", "JKCEMENT.NS", "JUBLPHARMA.NS", "KAJARIACER.NS", "LAURUSLABS.NS",
    "LTTS.NS", "LALPATHLAB.NS", "MFSL.NS", "MOTHERSON.NS",
    "MRF.NS", "MPHASIS.NS", "NAM-INDIA.NS", "NATIONALUM.NS", "NBCC.NS",
    "NLCINDIA.NS", "OBEROIRLTY.NS", "OFSS.NS", "PERSISTENT.NS", "PFC.NS",
    "RAIN.NS", "RAMCOCEM.NS", "SAIL.NS", "SOLARINDS.NS", "SONACOMS.NS",
    "STAR.NS", "SUPREMEIND.NS", "TATAELXSI.NS", "TORNTPOWER.NS", "ZEEL.NS",
]
