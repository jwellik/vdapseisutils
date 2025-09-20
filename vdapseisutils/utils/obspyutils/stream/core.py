"""
Core VStream and VTrace classes for volcano seismology workflows.

This module contains extended ObsPy Stream and Trace classes with additional
functionality specifically designed for volcano seismology data processing.

Classes
-------
VTrace : Extended Trace class for volcano seismology
VStream : Extended Stream class with preprocessing and data type management
"""

from obspy import Stream, Trace
import numpy as np
from typing import Optional, List, Tuple, Union, Any


class VTrace(Trace):
    """
    Extended Trace class for volcano seismology workflows.
    
    This class extends ObsPy's Trace class with additional functionality
    specifically designed for volcano seismology data processing.
    Currently serves as a base class for future volcano-specific trace methods.
    
    Inherits all functionality from obspy.Trace.
    """

    def __init__(self) -> None:
        """Initialize VTrace instance."""
        super().__init__()



class VStream(Stream):
    """
    Extended Stream class for volcano seismology workflows.
    
    This class extends ObsPy's Stream class with additional methods for
    volcano seismology data processing, including data type management
    and comprehensive preprocessing capabilities.
    
    Inherits all functionality from obspy.Stream.
    
    Methods
    -------
    set_data_type(dtype='int32') : Convert all trace data to specified numpy dtype
    preprocess(...) : Apply comprehensive preprocessing pipeline to all traces
    """

    def __init__(self, traces=None) -> None:
        """
        Initialize VStream instance.
        
        Parameters
        ----------
        traces : Stream, list of Trace, or None, optional
            Initial traces to populate the VStream. Can be:
            - An existing ObsPy Stream object
            - A list of ObsPy Trace objects  
            - None to create an empty VStream
            
        Examples
        --------
        >>> # Create empty VStream
        >>> vst = VStream()
        >>> 
        >>> # Create VStream from existing Stream
        >>> from obspy import read
        >>> st = read()
        >>> vst = VStream(st)
        >>> 
        >>> # Create VStream from list of traces
        >>> vst = VStream([trace1, trace2])
        """
        super().__init__(traces)

    def set_data_type(self, dtype: Union[str, np.dtype] = 'int32') -> 'VStream':
        """
        Convert all trace data to the specified numpy data type.
        
        This method ensures that all traces in the stream have the same data type,
        which is important for consistent processing. Also handles sampling rate
        rounding issues that can occur with different data sources.
        
        Returns
        -------
        VStream
            Returns self to allow method chaining.
        
        Parameters
        ----------
        dtype : str or numpy.dtype, default 'int32'
            The target numpy data type. Must be a valid numpy dtype.
            Common options: 'int32', 'float32', 'float64', etc.
            
        Raises
        ------
        TypeError
            If dtype is not a valid numpy data type.
            
        Examples
        --------
        >>> st = VStream()
        >>> st.set_data_type('float32')  # Convert all traces to float32
        >>> st.set_data_type(np.int16)   # Using numpy dtype object
        >>> # Method chaining example
        >>> st.set_data_type('float32').preprocess(resample=25.0)
        """
        # Validate the dtype parameter
        try:
            target_dtype = np.dtype(dtype)
        except TypeError:
            raise TypeError(f"Invalid numpy dtype: {dtype}. Must be a valid numpy data type.")
        
        for tr in self:
            # Convert data type if it doesn't match target
            if tr.data.dtype != target_dtype:
                tr.data = tr.data.astype(target_dtype)
            
            # Handle sampling rate rounding issues
            if tr.stats.sampling_rate != np.round(tr.stats.sampling_rate):
                tr.stats.sampling_rate = np.round(tr.stats.sampling_rate)
        
        return self

    def preprocess(self, 
                   resample: Optional[float] = None, 
                   taper: float = 5.0, 
                   filter: Optional[List[Union[str, dict]]] = None, 
                   trim: Optional[Tuple[Any, Any]] = None) -> 'VStream':
        """
        Apply basic pre-processing steps to all traces in the stream.
        
        This method performs common seismic data preprocessing operations including
        data type standardization, detrending, resampling, tapering, filtering, and trimming.
        All operations modify the stream and return self for method chaining.
        
        Parameters
        ----------
        resample : float, optional
            Target sampling rate in Hz. If provided, traces will be resampled
            to this rate if their current rate differs.
        taper : float, default 5.0
            Length in seconds to taper at the beginning and end of each trace
            before filtering to avoid edge effects.
        filter : list, optional
            Filter specification as [filter_type, filter_kwargs] where:
            - filter_type (str): Type of filter ('bandpass', 'lowpass', 'highpass', etc.)
            - filter_kwargs (dict): Filter parameters (e.g., {'freqmin': 1.0, 'freqmax': 10.0})
        trim : tuple, optional
            Time window (start_time, end_time) to trim all traces.
            Times can be UTCDateTime objects or relative times.
            
        Returns
        -------
        VStream
            Returns self to allow method chaining.
            
        Examples
        --------
        >>> st = VStream()
        >>> # Basic preprocessing with bandpass filter
        >>> st.preprocess(
        ...     resample=25.0,
        ...     filter=['bandpass', {'freqmin': 1.0, 'freqmax': 10.0}]
        ... )
        >>> 
        >>> # Preprocessing with trimming
        >>> from obspy import UTCDateTime
        >>> t1 = UTCDateTime('2023-01-01T00:00:00')
        >>> t2 = UTCDateTime('2023-01-01T01:00:00')
        >>> st.preprocess(trim=(t1, t2))
        >>> 
        >>> # Method chaining example
        >>> processed = st.preprocess(resample=25.0).merge(fill_value=0.0).filter('bandpass', freqmin=1.0, freqmax=10.0)
        """
        # Ensure consistent data types across all traces
        self.set_data_type()
        
        # Remove linear trends and mean
        self.detrend('demean')
        
        # Resample if requested
        if resample is not None:
            for tr in self:
                if tr.stats.sampling_rate != resample:
                    tr.resample(resample)
        
        # Apply tapering to avoid filter edge effects
        self.taper(max_percentage=None, max_length=taper)
        
        # Apply filter if specified
        if filter is not None:
            if not isinstance(filter, list) or len(filter) != 2:
                raise ValueError("Filter must be a list of [filter_type, filter_kwargs]")
            filter_type, filter_kwargs = filter
            self.filter(filter_type, **filter_kwargs)
        
        # Trim to specified time window
        if trim is not None:
            if not isinstance(trim, (list, tuple)) or len(trim) != 2:
                raise ValueError("Trim must be a tuple/list of (start_time, end_time)")
            self.trim(trim[0], trim[1])
        
        return self
