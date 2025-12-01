import numpy as np
import pandas as pd
from typing import Optional, Union
from .TS import *


# Frequency Series class
class fs(np.ndarray):
    """
    Frequency series object derived from FFT of a ts object.

    Attributes:
        delta_f (float): Frequency resolution.
        flen (int): Frequency domain length.
        tlen (int): Time domain length used for FFT.
        sampling_freq (float): Original sampling frequency.
        start (float): Start time of associated time series.
        _dynamic_attrs (dict): User-defined metadata (like .bp_meta, .ar_meta).
    """

    def __new__(
        cls,
        x: Union[np.ndarray, list],
        delta_f: float,
        flen: int,
        tlen: Optional[int] = None,
        sampling_freq: Optional[float] = None,
        start: Optional[float] = None,
    ) -> "fs":
        arr = np.asarray(x, dtype=np.complex128).view(cls)
        arr.delta_f = delta_f
        arr.flen = flen
        arr.tlen = tlen
        arr.sampling_freq = sampling_freq
        arr.start = start
        arr._dynamic_attrs = {}
        return arr

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.delta_f = getattr(obj, "delta_f", None)
        self.flen = getattr(obj, "flen", None)
        self.tlen = getattr(obj, "tlen", None)
        self.sampling_freq = getattr(obj, "sampling_freq", None)
        self.start = getattr(obj, "start", None)
        self._dynamic_attrs = getattr(obj, "_dynamic_attrs", {})

    def __getattr__(self, name):
        if name in self._dynamic_attrs:
            return self._dynamic_attrs[name]
        raise AttributeError(f"{name} not found")

    def __setattr__(self, name, value):
        if name in {
            "delta_f",
            "flen",
            "tlen",
            "sampling_freq",
            "start",
            "_dynamic_attrs",
        }:
            super().__setattr__(name, value)
        else:
            self._dynamic_attrs[name] = value

    def __dir__(self):
        return list(super().__dir__()) + list(self._dynamic_attrs.keys())

    def freqs(self) -> np.ndarray:
        """Return frequency axis based on delta_f and flen."""
        return np.arange(self.flen) * self.delta_f

    def duration(self) -> float:
        """Return duration of associated time series."""
        if self.tlen is None or self.sampling_freq is None:
            raise ValueError("Duration unavailable (missing tlen or sampling_freq).")
        return self.tlen / self.sampling_freq

    def to_df(self, val_name: str = "x") -> pd.DataFrame:
        """
        Convert to DataFrame with frequency and PSD columns.

        Args:
            val_name (str): Column name to assign to signal values. Default is "x".

        Returns:
            pd.DataFrame: DataFrame with 'freqs' and signal value column.
        """
        return pd.DataFrame({"freqs": self.freqs(), val_name: np.abs(self)})

    def to_ts(self) -> "ts":
        """
        Transform frequency series to time series via inverse FFT.

        Returns:
            ts: Time series object reconstructed from frequency domain.

        Raises:
            ValueError: If tlen or sampling_freq metadata is missing.
        """
        from .TS import ts, inherit_ts_attrs

        if self.tlen is None:
            raise ValueError(
                "Cannot convert to ts: tlen metadata is missing. "
                "This fs object may not have been created from a ts object."
            )

        if self.sampling_freq is None:
            raise ValueError(
                "Cannot convert to ts: sampling_freq metadata is missing. "
                "This fs object may not have been created from a ts object."
            )

        # Prepare FFT input with proper length
        fft_input = np.zeros(self.tlen // 2 + 1, dtype=np.complex128)
        fft_input[: self.flen] = self[: self.flen]

        # Perform inverse FFT and multiply by sampling_freq to reverse normalization
        time_data = np.fft.irfft(fft_input, n=self.tlen) * self.sampling_freq

        # Create ts object
        ts_obj = ts(
            time_data,
            start=self.start if self.start is not None else 0.0,
            sampling_freq=self.sampling_freq,
        )

        # Inherit any dynamic attributes from fs object
        inherit_ts_attrs(self, ts_obj)

        return ts_obj

    def plot(
        self,
        frange=None,
        logf=False,
        logy=False,
        title=None,
        lw=0.5,
        ax=None,
        figsize=(8, 4),
        **kwargs,
    ):
        """
        Plot the frequency series showing real and imaginary components.

        This is a convenience wrapper around plot_freq() that provides
        an object-oriented interface for plotting.

        Args:
            frange (tuple, optional): Frequency range (f_min, f_max) in Hz to display.
            logf (bool): Use logarithmic frequency axis. Default: False.
            logy (bool): Use logarithmic amplitude axis. Default: False.
            title (str, optional): Plot title.
            lw (float): Line width. Default: 0.5.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Creates new if None.
            figsize (tuple): Figure size if ax is None. Default: (8, 4).
            **kwargs: Additional keyword arguments passed to matplotlib plot().

        Returns:
            matplotlib.axes.Axes: The axes object containing the plot.

        Examples:
            >>> # Basic plot
            >>> fs_obj = ts_obj.to_fs()
            >>> ax = fs_obj.plot()

            >>> # With customization
            >>> ax = fs_obj.plot(frange=(10, 500), logf=True, logy=True)

            >>> # Plot multiple on same axes
            >>> fig, ax = plt.subplots()
            >>> fs1.plot(ax=ax, label="Signal 1")
            >>> fs2.plot(ax=ax, label="Signal 2")
            >>> ax.legend()
        """
        from .plot import plot_freq

        return plot_freq(
            self,
            frange=frange,
            logf=logf,
            logy=logy,
            title=title,
            lw=lw,
            ax=ax,
            figsize=figsize,
            **kwargs,
        )

    def __repr__(self) -> str:
        lines = [
            f"Frequency Series:",
            f"├─ delta_f = {self.delta_f}",
            f"├─ flen    = {self.flen}",
        ]
        if self.start is not None or self.sampling_freq is not None:
            lines += [
                f"└─ Associated ts info:",
                f"   ├─ Start time    = {self.start}",
                f"   └─ sampling.freq = {self.sampling_freq}",
            ]
        return "\n".join(lines) + "\n" + np.array2string(np.asarray(self), precision=4)


# Transform `ts` class to `fs` class
def to_fs(ts_obj: ts, delta_f: Optional[float] = None) -> fs:
    """
    Transform a ts object to a fs (frequency series) object via FFT.

    Args:
        ts_obj (ts): Input time series.
        delta_f (float, optional): Frequency resolution. Defaults to fs / len(ts).

    Returns:
        fs: Frequency series object with embedded ts metadata.
    """
    if delta_f is None:
        delta_f = ts_obj.sampling_freq / ts_obj.length

    tlen = int(round(1.0 / delta_f / (1 / ts_obj.sampling_freq) + 0.5))
    flen = int(round(tlen / 2) + 1)

    if tlen < ts_obj.length:
        raise ValueError(
            f"The value of delta_f={delta_f} results in undersampling. "
            f"Maximum delta_f is {1.0 / ts_obj.duration}"
        )

    padded = np.zeros(tlen, dtype=np.float64)
    padded[: ts_obj.length] = ts_obj.data

    # Normalize FFT by sampling_freq to match PyCBC convention
    fft_res = np.fft.rfft(padded) / ts_obj.sampling_freq

    fs_obj = fs(
        fft_res[:flen],
        delta_f=delta_f,
        flen=flen,
        tlen=tlen,
        sampling_freq=ts_obj.sampling_freq,
        start=ts_obj.start,
    )

    inherit_ts_attrs(ts_obj, fs_obj)

    return fs_obj
