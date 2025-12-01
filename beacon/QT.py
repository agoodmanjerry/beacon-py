# -------------------------------------------------------------------
# Q-transform implementation in Python
#
# This implementation is adapted from:
#   pycbc.filter.qtransform
#   (https://pycbc.org/pycbc/latest/html/_modules/pycbc/filter/qtransform.html)
#
# Original code is part of the PyCBC project,
# released under GNU General Public License v3.0 (GPL-3.0).
#
# This reimplementation follows the original logic but is rewritten
# for customized use in a modular Python workflow, originally reverse-translated from R.
#
# License: GPL-3.0-only
# -------------------------------------------------------------------

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from .Calc import amax, interp2d, cyclic
from .TS import *
from .FS import *

def deltam_f(mismatch: float) -> float:
    """
    Compute delta_m for Q-tiling from mismatch value.

    Args:
        mismatch (float): Fractional mismatch (e.g., 0.2)

    Returns:
        float: delta_m
    """
    return 2 * np.sqrt(mismatch / 3)


def iter_qs(qrange: Tuple[float, float], deltam: float) -> np.ndarray:
    """
    Generate a list of Q values spaced according to delta_m.

    Args:
        qrange (tuple): (min_q, max_q)
        deltam (float): Delta mismatch (from deltam_f)

    Returns:
        np.ndarray: Array of Q values
    """
    qmin, qmax = qrange
    cumum = np.log(qmax / qmin) / np.sqrt(2)
    nplanes = max(int(np.ceil(cumum / deltam)), 1)
    dq = cumum / nplanes

    qs = np.array([
        qmin * np.exp(np.sqrt(2) * dq * (i + 0.5))
        for i in range(nplanes)
    ])

    return qs

def iter_freqs(q: float, frange: Tuple[float, float], mismatch: float, duration: float) -> np.ndarray:
    """
    Generate a list of center frequencies for a given Q value.

    Args:
        q (float): Q value.
        frange (tuple): (min_freq, max_freq).
        mismatch (float): Fractional mismatch.
        duration (float): Duration of the original time series.

    Returns:
        np.ndarray: Array of center frequencies.
    """
    minf, maxf = frange
    cum_mismatch = np.log(maxf / minf) * np.sqrt(2 + q ** 2) / 2
    nfreq = max(1, int(np.ceil(cum_mismatch / deltam_f(mismatch))))
    fstep = cum_mismatch / nfreq
    fstep_min = 1 / duration

    qfrq = np.array([
        np.floor(
            (minf * np.exp(2 / np.sqrt(2 + q ** 2) * ((i + 0.5) * fstep))) / fstep_min
        ) * fstep_min
        for i in range(nfreq)
    ])

    return qfrq

def qtiling(fseries: fs,
            qrange: Tuple[float, float],
            frange: Tuple[float, float],
            mismatch: float = 0.2) -> List[Dict[str, Any]]:
    """
    Generate Q-tiling plan: list of dictionaries with Q and corresponding frequency tiles.

    Args:
        fseries (fs): Frequency series object.
        qrange (tuple): (min_q, max_q).
        frange (tuple): (min_freq, max_freq).
        mismatch (float): Tiling mismatch (default: 0.2)

    Returns:
        list of dict: Each element is {'q': Q, 'qfrq': list of f0}
    """
    duration = fseries.duration()
    qs = iter_qs(qrange, deltam_f(mismatch))

    qplane_tile_list = []
    for q in qs:
        qfrq = iter_freqs(q, frange, mismatch, duration)
        qplane_tile_list.append({"q": q, "qfrq": qfrq})

    return qplane_tile_list

def qseries(fseries: fs, 
            Q: float, 
            f0: float, 
            return_complex: bool = False) -> ts:
    """
    Extract Q-series filtered signal centered at f0 with quality factor Q.

    Args:
        fseries (fs): Frequency series object.
        Q (float): Quality factor.
        f0 (float): Center frequency.
        return_complex (bool): If True, return complex ts. Else, return normalized energy ts.

    Returns:
        ts: Time series with either complex values or normalized energy.
    """
    qprime = Q / np.sqrt(11)
    norm = np.sqrt(315 * qprime / (128 * f0))

    duration = fseries.duration()
    tlen = fseries.tlen
    fsamp = fseries.sampling_freq
    start = fseries.start

    window_size = 2 * int(f0 / qprime * duration) + 1
    fstart = int((f0 - (f0 / qprime)) * duration)
    fend = fstart + window_size

    # Clamp bounds
    fstart = max(0, fstart)
    fend = min(fend, tlen)

    fs_slice = fseries[fstart:fend]
    xfreqs = np.linspace(-1, 1, len(fs_slice))
    tapered = (1 - xfreqs**2)**2 * norm

    # Apply tapered window safely
    windowed = np.zeros(tlen, dtype=np.complex128)
    windowed[:len(fs_slice)] = fs_slice * tapered
    center = (fstart + fend) // 2
    windowed = cyclic(windowed, center)

    # IFFT to time domain
    ifft_res = np.fft.ifft(windowed)

    if return_complex:
        return ts(ifft_res, start=start, sampling_freq=fsamp)
    else:
        energy = np.abs(ifft_res) ** 2
        median_energy = np.median(energy)
        norm_energy = energy / median_energy if median_energy != 0 else energy
        return ts(norm_energy, start=start, sampling_freq=fsamp)

def qplane(qtile_list: List[Dict[str, Any]],
           fseries: fs,
           return_complex: bool = False) -> Dict[str, Any]:
    """
    Evaluate all Q tiles and select the plane with maximum energy.

    Args:
        qtile_list (list of dict): Each dict contains {'q': float, 'qfrq': array of f0s}
        fseries (fs): Frequency series.
        return_complex (bool): If True, return complex ts. Else, energy.

    Returns:
        dict: {'times': ..., 'freqs': ..., 'plane': 2D array}, with attr 'max_key'
    """
    max_energy = -np.inf
    max_key = -1
    max_q = None
    planes = []

    for i, tile in enumerate(qtile_list):
        q = tile['q']
        f0_list = tile['qfrq']
        energies = []

        for f0 in f0_list:
            ts_out = qseries(fseries, q, f0, return_complex=return_complex)
            energy_vals = np.abs(ts_out.data)**2 if return_complex else ts_out.data

            # Track max energy
            local_max = amax(energy_vals)
            if i == 0 or local_max > max_energy:
                max_energy = local_max
                max_key = i
                max_q = q
                energy_ts_ref = ts_out  # for time reference

            energies.append(energy_vals)

        # Stack over frequency axis
        plane_i = np.column_stack(energies)
        planes.append({"q": q, "energies": plane_i})

    plane = planes[max_key]['energies']
    freqs = qtile_list[max_key]['qfrq']
    times = energy_ts_ref.times

    result = {"times": times, "freqs": freqs, "plane": plane}
    result["_max_q"] = max_q  # Attach Q with max energy

    return result

def qtransform(ts_obj: ts,
               delta_t: Optional[float] = None,
               delta_f: Optional[float] = None,
               logfsteps: Optional[int] = None,
               frange: Optional[Tuple[float, float]] = None,
               qrange: Tuple[float, float] = (4, 64),
               mismatch: float = 0.2,
               return_complex: bool = False) -> Dict[str, Any]:
    """
    Compute Q-transform of a time series.

    Args:
        ts_obj (ts): Input time series.
        delta_t (float, optional): Desired time resolution for interpolation.
        delta_f (float, optional): Desired frequency resolution.
        logfsteps (int, optional): Log-spaced frequency steps (mutually exclusive with delta_f).
        frange (tuple, optional): Frequency range (min, max). Default: (30, fs/2 * 8).
        qrange (tuple): Q value range (default: (4, 64)).
        mismatch (float): Tiling mismatch (default: 0.2).
        return_complex (bool): If True, return complex time series. Else, normalized energy.

    Returns:
        dict: {'times': ..., 'freqs': ..., 'q_plane': ...}
    """
    if delta_f is not None and logfsteps is not None:
        raise ValueError("Provide only one of delta_f or logfsteps.")

    if frange is None:
        frange = (30, ts_obj.sampling_freq / 2)

    # Step 1: FFT
    fseries = to_fs(ts_obj)

    # Step 2: Q tiling
    qtiles = qtiling(fseries, qrange=qrange, frange=frange, mismatch=mismatch)

    # Step 3: Evaluate Q plane
    qres = qplane(qtiles, fseries, return_complex=return_complex)
    times = qres["times"]
    freqs = qres["freqs"]
    q_plane = qres["plane"]

    # Step 4: Optional interpolation
    do_interp = delta_t is not None or delta_f is not None or logfsteps is not None
    if do_interp:
        if delta_t is not None:
            interp_times = np.arange(ts_obj.start, ts_obj.end, delta_t)
        else:
            interp_times = times

        if delta_f is not None:
            interp_freqs = np.arange(frange[0], frange[1], delta_f)
        elif logfsteps is not None:
            interp_freqs = np.logspace(np.log10(frange[0]),
                                       np.log10(frange[1]),
                                       logfsteps)
        else:
            interp_freqs = freqs

        if return_complex:
            amp_interp = interp2d(times, freqs, np.abs(q_plane),
                                  xout=interp_times, yout=interp_freqs)['z']
            phase_interp = interp2d(times, freqs, np.angle(q_plane),
                                    xout=interp_times, yout=interp_freqs)['z']
            q_plane = amp_interp * np.exp(1j * phase_interp)
        else:
            q_plane = interp2d(times, freqs, q_plane,
                               xout=interp_times, yout=interp_freqs)['z']

        return {
            "times": interp_times,
            "freqs": interp_freqs,
            "q_plane": q_plane
        }

    # Step 5: No interpolation
    return {
        "times": times,
        "freqs": freqs,
        "q_plane": q_plane
    }
