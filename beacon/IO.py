# Load HDF5 GW data
import h5py
import numpy as np
import polars as pl
import requests
from .DQ import *
from .TS import *
from .etc import Rist

# Read GW data of hdf5 format (.hdf5, .h5)
def read_H5(file, sampling_freq, dq_level='all'):
    """
    Load HDF5 gravitational wave strain data and construct a time series object.

    Args:
        file (str): Full path to the HDF5 file.
        sampling_freq (float): Sampling frequency in Hz.
        dq_level (str, optional): Data quality (DQ) level to apply. Defaults to 'all'.

    Returns:
        ts: Time series object containing strain data.
            Attributes:
                meta (dict): Metadata dictionary extracted from the file.
                dqmask (pd.DataFrame, optional): Data quality mask indexed by GPS time.
                    Has attribute 'level' indicating the DQ level.
    """
    tmp = h5py.File(file, mode='r')

    strain = tmp['strain']['Strain'][()]
    meta_dict = {k: tmp['meta'][k][()] for k in tmp['meta'].keys()}
    tstart = meta_dict['GPSstart']

    # strain ts object
    tsobj = ts(strain, start=tstart, sampling_freq=sampling_freq)
    meta_Rist = Rist(meta_dict)
    
    # DQ mask
    if 'quality' in tmp.keys():
        dqmask = tmp['quality']['simple']['DQmask'][()]
        dt_dq = 1
        time_index = (tstart + np.arange(len(dqmask)) * dt_dq).astype(np.int64)

        dq_dicts = [dqlev(dq, level=dq_level) for dq in dqmask]

        dq_df = pl.DataFrame(dq_dicts).with_columns([
            pl.Series("t_floor", time_index)
        ])

        meta_Rist['DQ'] = Rist(
            level=dq_level,
            dqmask=dq_df
        )
        
    setattr(tsobj, 'meta', meta_Rist)

    tmp.close()
    del tmp
    
    return tsobj

# GWOSC information using API
# Download GWOSC catalog table
def get_gwosc(offline=False, csvpath=None) -> pd.DataFrame:
    """
    Retrieve the GWOSC event catalog as a pandas DataFrame.

    Args:
        offline (bool, optional): If True, load from a local CSV file instead of querying GWOSC.
        csvpath (str, optional): Path to the local CSV file (required if offline=True).

    Returns:
        pd.DataFrame: Flattened table of gravitational wave events, including strain parameters.
    """
    if offline:
        if csvpath is None:
            raise ValueError("CSV path must be provided in offline mode.")
        return pd.read_csv(csvpath)

    print("> Loading from GWOSC...")

    url = "https://www.gw-openscience.org/eventapi/jsonfull/query/show"
    response = requests.get(url)
    response.raise_for_status()
    query = response.json()["events"]  # dict: {event_id: event_dict}

    # 1. Bind rows
    records = []
    for full_id, v in query.items():
        base = v.copy()
        base["commonName"], base["v"] = full_id.rsplit("-", 1)
        base["id"] = full_id
        records.append(base)
    df = pd.DataFrame(records)

    # 2. Flatten strain
    def flatten_strain(s):
        if isinstance(s, dict):
            out = {}
            for k, v in s.items():
                if isinstance(v, dict):
                    if "value" in v:
                        out[f"{k}"] = v["value"]
                    if "lower" in v:
                        out[f"{k}_lower"] = v["lower"]
                    if "upper" in v:
                        out[f"{k}_upper"] = v["upper"]
                    if "unit" in v:
                        out[f"{k}_unit"] = v["unit"]
                else:
                    out[k] = v
            return pd.Series(out)
        else:
            return pd.Series()

    df_strain = df["strain"].apply(flatten_strain)

    # 3. Bind columns, drop 'strain', reorder
    df_final = pd.concat([df.drop(columns=["strain"]), df_strain], axis=1)

    return df_final

# Get values in specific rows (event names) and columns (parameter names)
def get_gwosc_param(gwosc_df, source_names, param):
    """
    Extract specific parameter values for given source(s) from the GWOSC DataFrame.

    Args:
        gwosc_df (pd.DataFrame): DataFrame returned by get_gwosc().
        source_names (str or list of str): Source common names to select.
        param (str or list of str): Column name(s) of the parameter(s) to retrieve.

    Returns:
        pd.Series or pd.DataFrame: Extracted parameter values.
            If a single parameter is requested, returns a Series;
            otherwise, returns a DataFrame.
    """
    # Ensure source_names is list-like
    if isinstance(source_names, str):
        source_names = [source_names]

    # Filter by source name
    df_filtered = gwosc_df[gwosc_df['commonName'].isin(source_names)]

    # Keep only the highest version per commonName
    df_latest = (
        df_filtered
        .sort_values('version', ascending=False)
        .drop_duplicates(subset='commonName', keep='first')
    )

    # Return based on param type
    if isinstance(param, str):
        return df_latest[param].squeeze()
    else:
        return df_latest[param]

