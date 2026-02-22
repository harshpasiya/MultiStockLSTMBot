import os
import yfinance as yf
import pandas as pd

def download_nse_data(symbols, period="3y"):
    """Download NSE stock data from yfinance"""

    os.makedirs("data", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading NSE data | period = {period}")
    print(f"{'='*60}\n")

    for symbol in symbols:
        try:
            print(f"Downloading {symbol}...", end=" ", flush=True)

            df = yf.download(
                symbol,
                period=period,
                progress=False,
                timeout=10,
                auto_adjust=True
            )

            # Quality check
            if df.empty or len(df) < 500:
                print(f"X Skip (only {len(df)} rows)")
                continue

            df = df.dropna()

            filename = f"data/{symbol.replace('.', '_')}_raw.csv"
            df.to_csv(filename)

            print(
                f"ok {len(df)} rows | "
                f"{df.index[0].date()} to {df.index[-1].date()}"
            )

        except Exception as e:
            print(f"Error: {str(e)}")

    print(f"\n{'='*60}")
    print("Data download completed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    primary_stocks = [
        # Power & Energy (high vol, momentum)
        "ADANIPOWER.NS", "TATAPOWER.NS", "JSWENERGY.NS", "NHPC.NS", "SJVN.NS",
        "TORNTPOWER.NS", "CESC.NS", "ADANIENSOL.NS", "NTPC.NS", "COALINDIA.NS",

        # Defence & Infra (momentum leaders)
        "HAL.NS", "BEL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "GRSE.NS",
        "IRCON.NS", "RVNL.NS", "BEML.NS", "Bharat Dynamics.NS", "DATA PATTERNS.NS",

        # Metals & Mining (high vol)
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "SAIL.NS", "NATIONALUM.NS",
        "VEDL.NS", "NMDC.NS", "MOIL.NS", "HINDCOPPER.NS", "APLAPOLLO.NS",

        # Banking & NBFC (midcap momentum)
        "BANKBARODA.NS", "PNB.NS", "UNIONBANK.NS", "CANBK.NS", "IOB.NS",
        "IDFCFIRSTB.NS", "RBLBANK.NS", "FEDERALBNK.NS", "KARURVYSYA.NS", "INDUSINDBK.NS",

        # Auto Ancillaries & EV (volatility)
        "EXIDEIND.NS", "MOTHERSUMI.NS", "BOSCHLTD.NS", "SAMVARDH.NS", "UNOMINDA.NS",
        "SUBROS.NS", "ENDURANCE.NS", "SUNDARMFAST.NS", "BALKRISIND.NS", "HDFCAMC.NS",

        # Chemicals & Speciality (momentum)
        "SRF.NS", "AARTIIND.NS", "DEEPAKNTR.NS", "ALKALI.NS", "TATACHEM.NS",
        "PIDILITIND.NS", "ASIANPAINT.NS", "INDIAMART.NS", "UPL.NS", "VINATIORGA.NS",

        # Logistics & Infra (high beta)
        "CONCOR.NS", "GMRINFRA.NS", "IRB.NS", "ADANIPORTS.NS", "JSWINFRA.NS",
        "PNCINFRA.NS", "KNRCON.NS", "HCC.NS", "DBL.NS", "L&T.NS",

        # Pharma & Healthcare (volatility)
        "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "METROPOLIS.NS",
        "POLICYBZR.NS", "LICI.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIGI.NS",

        # Consumer & Retail (midcap growth)
        "TRENT.NS", "DMART.NS", "JUBLFOOD.NS", "DEVYANI.NS", "WESTLIFE.NS",
        "PAGEIND.NS", "RAYMOND.NS", "MANYAVAR.NS", "ABFRL.NS", "VBL.NS",

        # IT & Tech (midcap momentum)
        "LTIM.NS", "MPHASIS.NS", "LTTS.NS", "COFORGE.NS", "PERSISTENT.NS",
        "KPITTECH.NS", "TATAELXSI.NS", "L&T TECH.NS", "HCLTECH.NS", "TECHM.NS",

        # Cement & Construction (volatility)
        "ULTRACEMCO.NS", "AMBUJACEM.NS", "ACC.NS", "DALBHARAT.NS", "JKCEMENT.NS",
        "DALMIA.NS", "SHREE CEMENT.NS", "GUJALKALI.NS", "NATIONALUM.NS", "GRASIM.NS",

        # Railways & Transport (momentum)
        "IRFC.NS", "RVNL.NS", "IRCON.NS", "TEXRAIL.NS", "IRCTC.NS",
        "CONCOR.NS", "TTML.NS", "GATI.NS", "DELHI VA.NS", "MAHINDRA LOG.NS",

        # Metals & Steel (high vol)
        "APOLLOPIPE.NS", "JINDALSTEL.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "SAIL.NS",
        "JSPL.NS", "GPIL.NS", "VSTLIFE.NS", "MINDACORP.NS", "POSCOIC.NS",

        # Auto & EV (momentum)
        "EXIDEIND.NS", "M&M.NS", "TATAMOTORS.NS", "MARUTI.NS", "EICHERMOT.NS",
        "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "ESCORTS.NS",

        # Fertilizer & Chemicals (volatility)
        "COROMANDEL.NS", "GNFC.NS", "MFL.NS", "DEEPAKFERT.NS", "PARAGMILK.NS",
        "UPL.NS", "SRF.NS", "PIDILITIND.NS", "AARTIIND.NS", "VINATIORGA.NS",

        # Cables & Wires (momentum)
        "POLYCAB.NS", "KEI.NS", "FINCABLES.NS", "STERLITE TECH.NS", "HFCL.NS",
        "PARAMOUNT COMM.NS", "PRECABLE.NS", "NEXANSINDIA.NS", "GENUSPOWER.NS", "INOXGREEN.NS",

        # Midcap Banks & Fin (high vol)
        "BANDHANBNK.NS", "YESBANK.NS", "KARURVYSYA.NS", "UCOBANK.NS", "INDIANB.NS",
        "CSBBANK.NS", "SOUTHBANK.NS", "EQUITASBNK.NS", "IDFCPB.NS", "DHANI.NS",

        # PSU & Defence (momentum)
        "BHEL.NS", "BEML.NS", "GRSE.NS", "MAZDOCK.NS", "HAL.NS",
        "BEL.NS", "Bharat Dynamics.NS", "DATA PATTERNS.NS", "PRESTIGE.NS", "L&T.NS",

        # Logistics & Real Estate (volatility)
        "INDIAGLYCOLS.NS", "INDIAMART.NS", "NAUKRI.NS", "INFO EDGE.NS", "JUSTDIAL.NS",
        "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS", "DELHIVERY.NS", "M&MFIN.NS",

        # Cement & Building Materials (momentum)
        "JKLAKSHMI.NS", "RAMCOCEM.NS", "VANDREVALA.NS", "GUJALKALI.NS", "DEEPAKNOHV.NS",
        "ASTRAZEN.NS", "ORIENTCEM.NS", "SANGHIIND.NS", "KANSAINER.NS", "PRSMJOHNSN.NS",

        # Auto Ancillaries (high vol)
        "BOSCHLTD.NS", "MOTHERSON.NS", "SAMVARDH.NS", "UNOMINDA.NS", "SUBROS.NS",
        "ENDURANCE.NS", "SUNDARMFAST.NS", "BALKRISIND.NS", "EXIDEIND.NS", "AMARAJABAT.NS",

        # Specialty Chemicals (momentum)
        "DEEPAKNTR.NS", "ALKALI.NS", "TATACHEM.NS", "PIDILITIND.NS", "ASIANPAINT.NS",
        "UPL.NS", "VINATIORGA.NS", "SRF.NS", "AARTIIND.NS", "LAXMIMACH.NS",

        # Midcap Consumer (volatility)
        "VBL.NS", "MARICO.NS", "DABUR.NS", "GODREJCP.NS", "COLPAL.NS",
        "BRITANNIA.NS", "NESTLEIND.NS", "HINDUNILVR.NS", "ITC.NS", "TATACONSUM.NS"
    ]

    download_nse_data(primary_stocks, period="5y")
    print("Next step: Run feature_engineering.py")