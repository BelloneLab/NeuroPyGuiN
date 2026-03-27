import numpy as np
import pandas as pd
import scipy.stats as stats
from collections import OrderedDict

from phylib.stats import correlograms

from ...common.epoch import Epoch
from ...common.utils import printProgressBar

def calculate_ibl_metrics(spike_times, spike_clusters, amplitudes, params, sample_rate, epochs = None):

    """ Calculate metrics for all units on one probe

    Inputs:
    ------
    spike_times : numpy.ndarray (num_spikes x 0)
        Spike times in seconds (same timebase as epochs)
    spike_clusters : numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    spike_templates : numpy.ndarray (num_spikes x 0)
        template IDs for each spike time
    amplitudes : numpy.ndarray (num_spikes x 0)
        Amplitude value for each spike time
    channel_map : numpy.ndarray (num_channels x 0)
        Original data channel for pc_feature_ind array
    channel_pos : numpy.ndarray (num_channels x 2)
        Original data channel positions in um
    templates : numpy.ndarray (num_units, num_timepoints, num_channels]
        Templates to which the spikes are assigned
    pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
    pc_feature_ind : numpy.ndarray (num_units x num_channels)
        Channel indices of PCs for each unit
    params : dict of parameters
        'isi_threshold' : minimum time for isi violations
        'tbin_sec' : time bin for ccg for contam_rate
    epochs : list of Epoch objects
        contains information on Epoch start and stop times

    
    Outputs:
    --------
    metrics : pandas.DataFrame
        one column for each metric
        one row per unit per epoch

    """

    metrics = pd.DataFrame()

    if epochs is None:
        epochs = [Epoch('complete_session', 0, np.inf)]
        
    
#   after any curation, the number of templates may not match the number of templates  
#   many of these cluster ID's may have no spikes assigned, 
#   but arrays of metrics need to be sized for the full set
    total_units = np.max(spike_clusters) + 1
    print('total unite: ' + repr(total_units))

    cluster_ids = np.arange(total_units)


    for epoch in epochs:

        in_epoch = (spike_times >= epoch.start_time) * (spike_times <= epoch.end_time)

        print("Calculating slidingRP")
        SRP_pass = calculate_slidingRP(spike_times[in_epoch], spike_clusters[in_epoch], total_units, sample_rate )
        
        print("Calculating nongaussian noise cutoff")
        nc_pass = calculate_noise_cutoff(np.squeeze(amplitudes[in_epoch]), spike_clusters[in_epoch], total_units)
                
        metrics = pd.concat((metrics, pd.DataFrame(data= OrderedDict((('cluster_id', cluster_ids),
                                ('slideingRP' , SRP_pass),
                                ('nongauss_noise_cutoff' , nc_pass),                                
                                )))))

    return metrics 



def calculate_slidingRP(spike_times, spike_clusters, total_units, sample_rate):

    cluster_ids = np.unique(spike_clusters)

    SRP_pass = np.zeros((total_units,))

    for idx, cluster_id in enumerate(cluster_ids):

        printProgressBar(idx+1, len(cluster_ids))

        for_this_cluster = (spike_clusters == cluster_id)
        SRP_pass[cluster_id] = slidingRP_viol(ts = spike_times[for_this_cluster], sample_rate = sample_rate) 

    return SRP_pass


def calculate_noise_cutoff(spike_amps, spike_clusters, total_units):

    cluster_ids = np.unique(spike_clusters)

    nc_pass = np.zeros((total_units,))

    for idx, cluster_id in enumerate(cluster_ids):

        printProgressBar(idx+1, len(cluster_ids))

        for_this_cluster = (spike_clusters == cluster_id)
        nc_pass[cluster_id] = noise_cutoff(amps = spike_amps[for_this_cluster])[0] 

    return nc_pass

def _max_acceptable_cont(FR, RP, rec_duration, acceptableCont, thresh):
    """
    Function to compute the maximum acceptable refractory period contamination
        called during slidingRP_viol
    """

    time_for_viol = RP * 2 * FR * rec_duration
    expected_count_for_acceptable_limit = acceptableCont * time_for_viol
    max_acceptable = stats.poisson.ppf(thresh, expected_count_for_acceptable_limit)
    if max_acceptable == 0 and stats.poisson.pmf(0, expected_count_for_acceptable_limit) > 0:
        max_acceptable = -1
    return max_acceptable


def slidingRP_viol(ts, bin_size=0.25, thresh=0.1, acceptThresh=0.1, sample_rate=30000):
    """
    A binary metric which determines whether there is an acceptable level of
    refractory period violations by using a sliding refractory period:

    This takes into account the firing rate of the neuron and computes a
    maximum acceptable level of contamination at different possible values of
    the refractory period. If the unit has less than the maximum contamination
    at any of the possible values of the refractory period, the unit passes.

    A neuron will always fail this metric for very low firing rates, and thus
    this metric takes into account both firing rate and refractory period
    violations.


    Parameters
    ----------
    ts : ndarray_like
        The timestamps (in s) of the spikes.
    bin_size : float
        The size of binning for the autocorrelogram, in msec.
    thresh : float
        Spike rate used to generate poisson distribution (to compute maximum
              acceptable contamination, see _max_acceptable_cont)
    acceptThresh : float
        The fraction of contamination we are willing to accept (default value
              set to 0.1, or 10% contamination)
    sample_rate : sample rate used in correlogram calculation

    Returns
    -------
    didpass : int
        0 if unit didn't pass
        1 if unit did pass

    See Also
    --------
    contamination

    Examples
    --------
    1) Compute whether a unit has too much refractory period contamination at
    any possible value of a refractory period, for a 0.25 ms bin, with a
    threshold of 10% acceptable contamination
        >>> ts = units_b['times']['1']
        >>> didpass = bb.metrics.slidingRP_viol(ts, bin_size=0.25, thresh=0.1,
                                                acceptThresh=0.1)
    """

    b = np.arange(0, 10.25, bin_size) / 1000 + 1e-6  # bins in seconds
    bTestIdx = [5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40]
    # with binSize = 0.25, these correspond to refractory periods of 
    # [1.25, 1.5, 1.75,2, 2.5, 3., 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]
    bTest = [b[i] for i in bTestIdx]

    if len(ts) > 0 and ts[-1] > ts[0]:  # only do this for units with samples
        recDur = (ts[-1] - ts[0])
        # compute acg
        c0 = correlograms(ts, np.zeros(len(ts), dtype='int8'), cluster_ids=[0],
                          bin_size=bin_size / 1000, sample_rate=sample_rate,
                          window_size=2,
                          symmetrize=False)
        # cumulative sum of acg, i.e. number of total spikes occuring from 0
        # to end of that bin
        cumsumc0 = np.cumsum(c0[0, 0, :])
        # cumulative sum at each of the testing bins
        res = cumsumc0[bTestIdx]
        total_spike_count = len(ts)

        # divide each bin's count by the total spike count and the bin size
        bin_count_normalized = c0[0, 0] / total_spike_count / bin_size * 1000
        num_bins_2s = len(c0[0, 0])  # number of total bins that equal 2 secs
        num_bins_1s = int(num_bins_2s / 2)  # number of bins that equal 1 sec
        # compute fr based on the  mean of bin_count_normalized from 1 to 2 s
        # instead of as before (len(ts)/recDur) for a better estimate
        fr = np.sum(bin_count_normalized[num_bins_1s:num_bins_2s]) / num_bins_1s
        mfunc = np.vectorize(_max_acceptable_cont)
        # compute the maximum allowed number of spikes per testing bin
        m = mfunc(fr, bTest, recDur, fr * acceptThresh, thresh)
        # did the unit pass (resulting number of spikes less than maximum
        # allowed spikes) at any of the testing bins?
        didpass = int(np.any(np.less_equal(res, m)))
    else:
        didpass = 0

    return didpass



def noise_cutoff(amps, quantile_length=.25, n_bins=100, nc_threshold=5, percent_threshold=0.10):
    """
    A new metric to determine whether a unit's amplitude distribution is cut off
    (at floor), without assuming a Gaussian distribution.
    This metric takes the amplitude distribution, computes the mean and std
    of an upper quartile of the distribution, and determines how many standard
    deviations away from that mean a lower quartile lies.
    Parameters
    ----------
    amps : ndarray_like
        The amplitudes (in uV) of the spikes.
    quantile_length : float
        The size of the upper quartile of the amplitude distribution.
    n_bins : int
        The number of bins used to compute a histogram of the amplitude
        distribution.
    n_low_bins : int
        The number of bins used in the lower part of the distribution (where
        cutoff is determined).
     nc_threshold: float
        the noise cutoff result has to be lower than this for a neuron to fail
    percent_threshold: float
        the first bin has to be greater than percent_threshold for neuron the to fail
    Returns
    -------
    cutoff : float
        Number of standard deviations that the lower mean is outside of the
        mean of the upper quartile.
    See Also
    --------
    missed_spikes_est
    Examples
    --------
    1) Compute whether a unit's amplitude distribution is cut off
        >>> amps = spks_b['amps'][unit_idxs]
        >>> cutoff = bb.metrics.noise_cutoff(amps, quantile_length=.25, n_bins=100)
    """
    cutoff = np.float64(np.nan)
    first_low_quantile = np.float64(np.nan)
    fail_criteria = np.ones(1).astype(bool)[0]

    if amps.size > 1:  # ensure there are amplitudes available to analyze
        bins_list = np.linspace(0, np.max(amps), n_bins)  # list of bins to compute the amplitude histogram
        n, bins = np.histogram(amps, bins=bins_list)  # construct amplitude histogram
        idx_peak = np.argmax(n)  # peak of amplitude distribution
        # don't count zeros #len(n) - idx_peak, compute the length of the top half of the distribution -- ignoring zero bins
        length_top_half = len(np.where(n[idx_peak:-1] > 0)[0])
        # the remaining part of the distribution, which we will compare the low quantile to
        high_quantile = 2 * quantile_length
        # the first bin (index) of the high quantile part of the distribution
        high_quantile_start_ind = int(np.ceil(high_quantile * length_top_half + idx_peak))
        # bins to consider in the high quantile (of all non-zero bins)
        indices_bins_high_quantile = np.arange(high_quantile_start_ind, len(n))
        idx_use = np.where(n[indices_bins_high_quantile] >= 1)[0]

        if len(n[indices_bins_high_quantile]) > 0:  # ensure there are amplitudes in these bins
            # mean of all amp values in high quantile bins
            mean_high_quantile = np.mean(n[indices_bins_high_quantile][idx_use])
            std_high_quantile = np.std(n[indices_bins_high_quantile][idx_use])
            if std_high_quantile > 0:
                first_low_quantile = n[(n != 0)][1]  # take the second bin
                cutoff = (first_low_quantile - mean_high_quantile) / std_high_quantile
                peak_bin_height = np.max(n)
                percent_of_peak = percent_threshold * peak_bin_height

                fail_criteria = (cutoff > nc_threshold) & (first_low_quantile > percent_of_peak)

    nc_pass = ~fail_criteria
    return nc_pass, cutoff, first_low_quantile