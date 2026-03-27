# NeuroPyxels `npyx` Analysis Function Inventory

Auto-generated from source using AST. Descriptions come from function docstrings when present; otherwise from function names/signatures.

## `behav.py`

- `behav_dic`: Remove artefactual licking, processes rotary encoder and wheel turn trials.
- `npix_aligned_paq`: Aligns thresholded paqIO data at f_behav to npix data at dp.
- `load_PAQdata`: Used to load analog (wheel position...)
- `get_wheelturn_df_dic`: Arguments:
- `load_baseline_periods`: Function to calculate periods of undisturbed neural activity
- `align_variable`: Arguments:
- `fast_histogram`: Function `fast_histogram(values, bins)`.
- `fast_histogram_weights`: Function `fast_histogram_weights(values, bins, weights)`.
- `process_all_events`: Function `process_all_events(events, variable_t, variable, window_s_0, window_s_1, tbins)`.
- `align_variable_numba`: Arguments:
- `align_times`: Arguments:
- `align_times_manyevents`: Will run faster than align_times if many events are provided (will run in approx 800ms for 10 or for 600000 events,
- `fast_align_times`: Arguments:
- `process_all_events_times`: Function `process_all_events_times(times, events, window_0, window_1, tbins)`.
- `align_times_numba`: Arguments:
- `jPSTH`: From A. M. H. J. AERTSEN, G. L. GERSTEIN, M. K. HABIB, AND G. PALM, 1989, Journal of Neurophysiology
- `trial_selector`: Given a set of events and trains (in samples),
- `get_ifr`: Arguments:
- `process_2d_trials_array`: Function `process_2d_trials_array(y, y_bsl, zscore, zscoretype, convolve, gsd...)`.
- `get_processed_ifr`: Returns the "processed" (averaged and/or smoothed and/or z-scored) instantaneous firing rate of a neuron.
- `get_processed_BTN_matrix`: Returns a matrix M of shape (B,T,N) = (n bins, n trials, n neurons).
- `get_BTN_matrix`: B: n bins, T: n trials, N: n neurons
- `get_poisson_BTN_matrix`: Function `get_poisson_BTN_matrix(M)`.
- `filter_allneurons_active`: - M: BxTxN matrix
- `get_processed_popsync`: Arguments:
- `psth_fraction_pop_sync`: Computes the population synchrony for a set of events.
- `decode_rotary`: Function to decode velocity from rotary encoder channels.
- `get_nframes`: Function `get_nframes(video_path)`.
- `frameid_vidpath`: Return relative frame id and respective video
- `frame_from_vid`: Function `frame_from_vid(video_path, frame_i, plot)`.
- `cart2pol`: Arguments:
- `pol2cart`: Arguments:
- `get_polar_vect`: Arguments:
- `ellipsis`: - a, b: floats, length of horizontal/vertical axis (for rot=0), respectively
- `in_ellipsis`: Function `in_ellipsis(X, Y, a, b, x0, y0...)`.
- `ellipsis_string`: - x: float, x (or y) coordinate of string on axis 'axis' (mm)
- `draw_wheel_mirror`: Homologous to a cylindrical wedge (plane crossing <1 basis of cylinder).

## `c4/acg_augmentations.py`

- `fast_acg3d`: Function `fast_acg3d(spike_times, win_size, bin_size, fs, num_firing_rate_bins, smooth...)`.

## `c4/acg_vs_firing_rate.py`

- `fast_acg3d`: Function `fast_acg3d(spike_times, win_size, bin_size, fs, num_firing_rate_bins, smooth...)`.
- `delete_spikes`: Function `delete_spikes(spikes, deletion_prob)`.
- `add_spikes`: Function `add_spikes(spikes, max_addition)`.
- `random_jitter`: Function `random_jitter(spikes, max_shift)`.
- `augment_spikes`: Function `augment_spikes(spikes_list, *transforms)`.
- `aux_compute_acgs`: Function `aux_compute_acgs(spikes, win_size, bin_size, sampling_rate, fast, i)`.

## `c4/dataset_init.py`

- `xor_decrypt`: Function `xor_decrypt(encrypted_data, password)`.
- `download_file`: Function `download_file(url, output_path, description, requires_password)`.
- `save_results`: Function `save_results(results_dict, save_path)`.
- `combine_features`: Function `combine_features(df1, df2)`.
- `get_paths_from_dir`: Function `get_paths_from_dir(path_to_dir, include_lisberger, include_medina, include_hull_unlab)`.
- `extract_and_merge_datasets`: Function `extract_and_merge_datasets(*dataset_paths)`.
- `make_dataset_df`: Function `make_dataset_df(dataset)`.
- `extract_and_check`: Function `extract_and_check(*dataset_paths)`.
- `prepare_classification_dataset`: Function `prepare_classification_dataset(dataset, normalise_acgs, win_size, bin_size, multi_chan_wave, process_multi_channel...)`.
- `plot_quality_checks`: Function `plot_quality_checks(dataframe, lab, fig_title)`.
- `save_quality_plots`: Function `save_quality_plots(dataframe, save_folder, unlabelled, lisberger)`.
- `save_and_close`: Function `save_and_close(merged_df, lab, fig_title, file_name)`.
- `save_acg`: Function `save_acg(spike_train, unit_n, save_name)`.
- `quality_failed_plot`: Function `quality_failed_plot(save_name)`.
- `save_wvf`: Function `save_wvf(waveform, save_name)`.
- `make_summary_plots_wvf`: Function `make_summary_plots_wvf(dataset, save_folder, monkey)`.
- `make_summary_plots_preprocessed_wvf`: Function `make_summary_plots_preprocessed_wvf(dataset, save_folder, monkey)`.
- `make_summary_plots_wvf_by_line`: Function `make_summary_plots_wvf_by_line(dataset, save_folder)`.
- `make_summary_plots_acg_by_line`: Function `make_summary_plots_acg_by_line(dataset, save_folder)`.
- `make_summary_plots_acg`: Function `make_summary_plots_acg(dataset, save_folder, monkey)`.
- `make_raw_plots`: Function `make_raw_plots(dataset, path_to_dir, folder)`.
- `find_unusable_index`: Function `find_unusable_index(feat_df, dataset_df, unusable_features_idx)`.
- `report_unusable_features`: Function `report_unusable_features(feat_df, dataset_df, unusable_features_idx, args, save, lisberger)`.
- `save_features`: Function `save_features(feat_df, features_name, args, bad_idx, drop_cols, monkey)`.
- `make_plots_folder`: Function `make_plots_folder(args)`.
- `summary_plots`: Function `summary_plots(args, dataset_class, by_line, raw, monkey)`.
- `calc_snr`: Function `calc_snr(wvf, noise_samples, return_db)`.

## `c4/dl_transforms.py`

- `fixmatch_augment_pool`: Function `fixmatch_augment_pool()`.
- `waveform_augment_pool`: Function `waveform_augment_pool()`.

## `c4/dl_utils.py`

- `define_forward_vae`: Function `define_forward_vae(in_features, init_weights, params, device)`.
- `load_acg_vae`: Function `load_acg_vae(encoder_path, win_size, bin_size, d_latent, initialise, device...)`.
- `load_waveform_encoder`: Function `load_waveform_encoder(encoder_args, encoder_path, in_features, initialise, device)`.
- `load_waveform_vae`: Function `load_waveform_vae(encoder_args, encoder_path, device)`.
- `ELBO_VAE`: Computes the Evidence Lower Bound (ELBO) for a Variational Autoencoder (VAE).
- `generate_kl_weight`: Generate an array of weights to be used for the KL divergence loss in a VAE model.

## `c4/encode_features.py`

- `encode_acgs`: Function `encode_acgs(encoder_path, acgs_3d, dataset)`.
- `encode_waveforms`: Function `encode_waveforms(encoder_path, waveforms, dataset, args)`.

## `c4/misc.py`

- `fix_autoreload`: Fix autoreload for decorators so decorated functions properly autoreload
- `require_advanced_deps`: Function `require_advanced_deps(*deps)`.

## `c4/monkey_dataset_init.py`

- `get_lisberger_dataset`: Function `get_lisberger_dataset(data_path)`.

## `c4/plots_functions.py`

- `make_plotting_df`: Function `make_plotting_df(df, save, save_path)`.
- `save_acg`: Function `save_acg(spike_train, unit_n, save_name)`.
- `quality_failed_plot`: Function `quality_failed_plot(save_name)`.
- `save_wvf`: Function `save_wvf(waveform, save_name)`.
- `save_amplitudes`: Function `save_amplitudes(times, amplitudes, dpi, save_name)`.
- `plot_confusion_from_proba`: Plot confusion matrix from model predictions and true targets.
- `plot_results_from_threshold`: Function `plot_results_from_threshold(true_targets, predicted_proba, correspondence, threshold, model_name, kde_bandwidth...)`.
- `plot_collapsed_densities`: Function `plot_collapsed_densities(all_true_positives, all_false_positives, kde_bandwidth, ax)`.
- `plot_cosine_similarity`: It takes a matrix of features, a vector of labels, and a dictionary that maps labels to cell types,
- `threshold_predictions`: Function `threshold_predictions(features, predicted_probabilties, threshold)`.
- `plot_waveforms`: Function `plot_waveforms(waveforms, title, col, central_range, ax)`.
- `get_relevant_waveform`: Function `get_relevant_waveform(wf, n_channels, central_range)`.
- `plot_acgs`: Function `plot_acgs(acgs, title, col, win_size, bin_size, ax)`.
- `plot_features_1cell_vertical`: Function `plot_features_1cell_vertical(i, acg_3ds, waveforms, predictions, saveDir, fig_name...)`.
- `plot_survival_confidence`: Function `plot_survival_confidence(confidence_matrix, correspondence, ignore_below_confidence, saveDir, correspondence_colors, use_confidence_ratio)`.

## `c4/predict_cell_types.py`

- `get_n_cores`: Function `get_n_cores(num_cores)`.
- `redirect_stdout_fd`: Function `redirect_stdout_fd(file)`.
- `handle_outdated_model`: Function `handle_outdated_model(exc_type, model_type)`.
- `directory_checks`: Function `directory_checks(data_path)`.
- `prepare_dataset_from_binary`: Function `prepare_dataset_from_binary(dp, units, again, fp_threshold, fn_threshold, peak_sign)`.
- `get_layer_information`: Function `get_layer_information(args, good_units)`.
- `prepare_dataset_from_h5`: Function `prepare_dataset_from_h5(data_path)`.
- `aux_prepare_dataset`: Function `aux_prepare_dataset(dp, u, again, fp_threshold, fn_threshold, peak_sign)`.
- `prepare_dataset_from_binary_parallel`: Function `prepare_dataset_from_binary_parallel(dp, units, again, fp_threshold, fn_threshold, peak_sign)`.
- `prepare_dataset`: Prepare the dataset for classification.
- `format_predictions`: Formats the predictions matrix by computing the mean predictions, prediction confidences, delta mean confidences,
- `save_serialised`: Function `save_serialised(la, filepath)`.
- `load_serialised`: Function `load_serialised(filepath)`.
- `load_precalibrated_ensemble`: Function `load_precalibrated_ensemble(models_directory, fast)`.
- `save_calibrated_ensemble`: Function `save_calibrated_ensemble(calibrated_models, save_directory)`.
- `run_cell_types_classifier`: Predicts the cell types of units in a given dataset using a pre-trained ensemble of classifiers.
- `run_c4`: Function `run_c4()`.

## `c4/run_baseline_classifier.py`

- `train_predict_test`: Trains a classifier on the training data, predicts on the test data, and returns the f1 score and predicted probabilities.
- `plot_feature_importance`: Function `plot_feature_importance(feature_importance_list, features_dataframe, save_folder)`.
- `filter_out_granule_cells`: Function `filter_out_granule_cells(features, targets, return_dicts)`.
- `save_predictions_df`: Function `save_predictions_df(args, dataset_info, results_dict, save_path, repeats, probability_type...)`.
- `get_model_class`: Function `get_model_class(args)`.

## `c4/run_deep_classifier.py`

- `download_vaes`: Function `download_vaes()`.
- `remove_dropout`: Removes or disables dropout layers in the given PyTorch model.
- `plot_training_curves`: Function `plot_training_curves(train_losses, f1_train, epochs, save_folder)`.
- `calculate_accuracy`: Function `calculate_accuracy(y_pred, y)`.
- `train`: Function `train(model, iterator, optimizer, criterion, device)`.
- `layer_correction`: Applies corrections to the predicted probabilities based on the layer information.
- `get_kronecker_hessian_attributes`: Function `get_kronecker_hessian_attributes(*kronecker_hessians)`.
- `predict_unlabelled`: Predicts the probabilities of the test set using a Laplace model.
- `get_model_probabilities`: Computes the probabilities of a given model for a test dataset, with or without Laplace approximation calibration.
- `define_transformations`: Function `define_transformations(norm_acg, log_acg, transform_acg, transform_wave, multi_chan)`.
- `save_ensemble`: Function `save_ensemble(models_states, file_path)`.
- `load_ensemble`: Function `load_ensemble(file_path, device, pool_type, n_classes, use_layer, fast...)`.
- `load_calibrated_ensemble`: Function `load_calibrated_ensemble(models, hessians)`.
- `ensemble_predict`: Function `ensemble_predict(ensemble, test_iterator, device, method, enforce_layer, labelling)`.
- `cross_validate`: Function `cross_validate(dataset, targets, spikes, acg_vae_path, args, layer_info...)`.
- `plot_confusion_matrices`: Function `plot_confusion_matrices(results_dict, save_folder, model_name, labelling, correspondence, plots_prefix...)`.
- `ensemble_inference`: Performs inference on a test dataset using an ensemble of models.
- `post_hoc_layer_correction`: Function `post_hoc_layer_correction(results_dict, one_hot_layer, labelling, repeats)`.
- `encode_layer_info_original`: Function `encode_layer_info_original(layer_information)`.
- `encode_layer_info`: Function `encode_layer_info(layer_information)`.

## `circuitProphyler.py`

- `ask_syncchan`: Function `ask_syncchan(ons)`.

## `corr.py`

- `make_phy_like_spikeClustersTimes`: - trains: list of spike trains, in samples.
- `make_matrix_2xNevents`: Arguments:
- `crosscorrelate_cyrille`: Returns the crosscorrelation function of two spike trains.
- `crosscorr_cyrille`: Computes crosscorrelation histograms between all pairs of neuron spike trains.
- `get_log_bins_samples`: log_window_end in ms, fs is sampling rate - output in samples.
- `ccg`: ********
- `ccg_hz`: Shorthand to get the ccg in Hertz,
- `ccg_2d_numba`: Compute the (n_events, n_bins) crosscorrelogram between two time series,
- `ccg_2d`: Compute the (n_events, n_bins) crosscorrelogram between two time series.
- `acg`: ********
- `scaled_acg`: - get the spike times passing our quality metric from the first 20 mins
- `acg_3D`: Function `acg_3D(dp, u, cbin, cwin, normalize, verbose...)`.
- `ccg_3D`: Function `ccg_3D(dp, U, cbin, cwin, normalize, verbose...)`.
- `crosscorr_vs_firing_rate`: Computes a "three dimensional" cross-correlogram that shows firing regularity when the neuron is
- `ccg_vs_fr`: Computes a "three dimensional" cross-correlogram that shows firing regularity when the neuron is
- `convert_acg_log`: Interpolates an autocorrelogram computed with linear bins on a log-scale
- `get_ccgstack_fullname`: Function `get_ccgstack_fullname(name, cbin, cwin, normalize, periods)`.
- `ccg_stack`: Routine generating a stack of correlograms for faster subsequent analysis,
- `compute_ccgs_bulk`: Function `compute_ccgs_bulk(ccg_inputs, parallel)`.
- `get_ustack_i`: Finds indices of units inside a ccg stack.
- `canUse_Nbins`: Function to assess the number of expected triplets (3 consecutive bins) in a crosscorrelogram.
- `KopelowitzCohen2014_ccg_significance`: Function to assess whether a correlogram is significant or not.
- `StarkAbeles2009_ccg_sig`: Predictor and p-values for CCG using convolution.
- `StarkAbeles2009_ccg_significance`: Arguments:
- `get_cross_features`: Returns features of a correlogram modulation as defined in Kopelowitz et al. 2014.
- `get_ccg_sig`: Arguments:
- `ccg_sig_stack`: Arguments:
- `gen_sfc`: Function generating a functional correlation dataframe sfc (Nsig x 2+8 features) and matrix sfcm (Nunits x Nunits)
- `cisi_numba_para`: Function `cisi_numba_para(spk1, spk2, available_memory)`.
- `cisi_numba`: Function `cisi_numba(spk1, spk2, available_memory)`.
- `cisi_chunk`: Function `cisi_chunk(i, chunk, n, spk2, direction, s)`.
- `get_cisi`: Computes cross spike intervals i.e time differences between
- `par_process`: Function `par_process(i, chunk, spk2, n, direction)`.
- `get_cisi_parprocess`: Computes cross spike intervals i.e time differences between
- `pearson_corr`: Calculate the NxN matrix of pairwise Pearsonâ€™s correlation coefficients
- `pearson_corr_trn`: Calculate the NxN matrix of pairwise Pearsonâ€™s correlation coefficients
- `correlation_index`: Calculate the NxN matrix of pairwise correlation indices from Wong, Meister and Shatz 1993
- `synchrony_regehr`: - CCG: crosscorrelogram array, units does not matter. Should be long enough.
- `synchrony`: Function `synchrony(CCG, cbin, sync_win, fract_baseline)`.
- `synchrony_zscore`: - CCG: crosscorrelogram array, units does not matter. Should be longer than sync_win obviously.
- `synchrony_deltaproba`: - CCG: crosscorrelogram array, units in probability change. Should be longer than sync_win obviously.
- `covariance`: Simply computes covariance according to the formula:
- `convert_ccg_to_covariance`: Converts a crosscorrelogram in probability to covariance values
- `cofiring_tags`: Returns a boolean array of len of train of t.
- `frac_pop_sync_old`: Returns an array of size len(t1),
- `frac_pop_sync`: Returns an array of size len(t1),
- `fraction_pop_sync`: Wrapper for frac_pop_sync:
- `get_cm`: Make correlation matrix.
- `spike_time_tiling_coefficient`: Calculates the Spike Time Tiling Coefficient (STTC) as described in
- `PSDxy`: ********

## `datasets.py`

- `save`: Function `save(file_name, obj)`.
- `load`: Function `load(file_name)`.
- `get_neuron_attr`: Prompts the user to select a given neuron's file to load.
- `get_neuron_attr_generic`: Function `get_neuron_attr_generic(neuron_ids, pi, hdf5_file)`.
- `ls`: Given an hdf5 file path or an open hdf5 file python object, returns the child directories.
- `normalise_wf`: Custom normalisation so that the through of the waveform is set to -1
- `crop_original_wave`: It takes a waveform of shape (n_channels, central_range) and returns a copy of
- `crop_chanmap`: Function `crop_chanmap(chanmap, peak_channel_idx, n_channels)`.
- `resample_acg`: Given an ACG, add artificial points to it.
- `get_h5_absolute_ids`: Function `get_h5_absolute_ids(h5_path)`.
- `decode_string`: The function decodes a given value to a string if it is of type bytes or numpy bytes, and returns
- `process_label`: Function `process_label(label)`.
- `merge_h5_datasets`: Merges multiple NeuronsDatasets instances into one
- `resample_waveforms`: It takes a dataset, resizes the waveforms to a new sampling rate, and returns a new dataset with the
- `force_amplitudes_length`: Function `force_amplitudes_length(amplitudes, times)`.
- `preprocess_template`: This function preprocesses a given template by resampling it, aligning it to a peak, flipping it if
- `preprocess_template_singlewaveforms`: This function generalizes npyx.datasets.preprocess_template to (n_samples, n_waveforms) and (n_channels, n_samples, n_waveforms) arrays. It computes the preprocessing parameters from the average across n_waveforms (like npyx.datasets.preprocess_template), but applies the preprocessing to the original full array.
- `pad_matrix_with_decay`: Function `pad_matrix_with_decay(matrix, target_channels)`.

## `feat.py`

- `acg_burst_vs_mfr`: It computes the autocorrelogram of the spike train, smooths it, and then computes the ratio of the
- `compute_isi`: Input: spike times in samples and returns ISI of spikes that pass through
- `burst_index`: Calculate the burst index of a given inter-spike interval (ISI) array.
- `entropy_log_isi`: It takes a list of interspike intervals, bins them logarithmically, smooths the resulting histogram,
- `compute_isi_features`: `compute_isi_features` takes a list of interspike intervals (ISIs) and returns a list of features
- `cross_zero_t`: Find the first time that the waveform crosses zero between t1 and t2, and return the time and the
- `plot_debug_peaks`: It plots the waveform, the margin, the onset and offset, and the peaks.
- `find_repolarisation`: If the minimum value is the last value, or the value after the minimum is not positive, then find the
- `detect_peaks`: Custom peak detection algorithm. Based on scipy.signal.find_peaks.
- `wvf_width`: The function `wvf_width` takes a waveform, a peak time, and a trough time, and returns the width
- `pt_ratio`: Calculates the absolute value of the peak to trough ratio.
- `trough_onset_t`: It finds the last time before the trough that the waveform crossed a threshold of 5% of the trough
- `peak_offset_t`: It finds the last time after the peak when the waveform relaxed to 5% of the peak value.
- `repol_10_90_t`: Find the 10th and 90th percentile values of the upslope of the waveform, then find the closest
- `depol_10_90_t`: Find the points where the waveform crosses 10% and 90% of the trough value, and return the time
- `depol_slope`: It fits a line to the downslope of the trough (from the half width),
- `pos_half_width`: Find the repolarisation half width (in time) of the waveform.
- `neg_half_width`: Find the depolarisation half width (in time) of the waveform.
- `tau_end_slope`: It fits an exponential to the end of the waveform, and returns the fit, the mean squared error, and
- `interp_wave`: It takes a waveform and interpolates it by a factor of `multi` along the `axis` dimension
- `repol_slope`: It fits a line to the upslope from the trough (up to the half width),
- `recover_chanmap`: Given a 1D array containing an incomplete channelmap (of only x coordinates),
- `dendritic_component`: Function `dendritic_component(waveform_2d, peak_chan, somatic_mask)`.
- `chan_spread`: It takes in the waveforms, the peak channel, and the channel map and returns the ratio of the mean
- `healthy_waveform`: Determine if waveform looks healthy.
- `is_somatic`: Assures that the waveform is somatic and can be used in further processing.
- `detect_peaks_2d`: For each channel, we detect peaks in the waveform, check if it has an healthy shape, and
- `filter_out_waves`: Filter out waveforms that are not within a certain range of the peak channel and have a
- `find_relevant_waveform`: If there are any somatic waveforms, return the best one. Otherwise, if there are any dendritic
- `find_relevant_peaks`: Given two arrays of peak times and peak values from detect_peaks,
- `extract_single_channel_features`: It takes a waveform and returns a list of features that describe the waveform
- `extract_spatial_features`: - peak_chan: channel from which the 1D waveworm features will be extracted
- `waveform_features`: > Given a 2D array of waveforms, the channel with the peak, and a boolean flag for whether to plot
- `waveform_features_json`: Given a path to a recording and a unit number, return a list of features for that unit.
- `plot_all_features`: Function `plot_all_features(waveform, normalise, dp, unit, label)`.
- `temporal_features`: It takes a list of spike times for each neuron, and returns a list of features that describe the
- `temporal_features_wrap`: High level function for getting the temporal features of a unit from a dataset at dp.
- `check_json_file`: It checks that the files and units specified in the json file exist
- `feature_extraction_json`: It takes a json file containing paths to all the recordings and extracts the features.
- `h5_feature_extraction`: It takes a NeuronsDataset instance coming from an h5 dataset and extracts the features.
- `get_unusable_features`: Returns the index of unusable features
- `prepare_classification`: Prepares the dataframe for classification.

## `gl.py`

- `get_npyx_memory`: Function `get_npyx_memory(dp)`.
- `get_datasets`: Function to load dictionnary of dataset paths and relevant units.
- `json_connected_pairs_df`: Function `json_connected_pairs_df(ds_master, ds_paths_master, ds_behav_master)`.
- `make_connected_pairs_df`: Function `make_connected_pairs_df(ds_master, ds_paths_master, ds_behav_master, upsample_sync, pval_th, sync_win...)`.
- `get_rec_len`: returns recording length in seconds or samples
- `detect_new_spikesorting`: Detects whether a dataset has been respikesorted
- `save_qualities`: Function `save_qualities(dp, qualities)`.
- `generate_units_qualities`: Creates an empty table of units qualities ("groups" as in good, mua,...).
- `load_units_qualities`: Load unit qualities (groups tsv table) from dataset.
- `load_merged_units_qualities`: Load unit qualities from merged dataset.
- `get_units`: Function `get_units(dp, quality, chan_range, again)`.
- `get_good_units`: Function `get_good_units(dp)`.
- `check_periods`: Function `check_periods(periods)`.
- `export_new_trains_to_phy`: Function to embed arbitrary spikes trains into a phy-compatible dataset as new units.

## `h5.py`

- `label_optotagged_unit_h5`: Add optotagged label to neuron.
- `reset_optotagged_labels`: Resets all optotagged labels to 0
- `add_unit_h5`: Adds a spike-sorted unit to a new or existing HDF5 five file using the
- `add_json_datasets_to_h5`: Wrapper function to loop over all datasets in a json file
- `add_json_datasets_to_h5_hausser`: Function `add_json_datasets_to_h5_hausser(json_path, h5_path, include_all_good, selective_overwrite, overwrite_h5)`.
- `load_json_datasets`: Function `load_json_datasets(json_path, include_missing_datasets)`.
- `add_units_to_h5`: Add all or specified units at the respective data path to an HDF5 file.
- `add_data_to_unit_h5`: Add data to neuron already in h5 file.
- `get_unit_paths_h5`: Function `get_unit_paths_h5(h5_file, dataset, unit, lab_id, unit_absolute_id)`.
- `remove_unit_h5`: Function `remove_unit_h5(h5_path, dp, unit, lab_id, dataset)`.
- `get_absolute_neuron_ids`: Function `get_absolute_neuron_ids(h5_path, again)`.
- `get_neuron_id_dict`: Function `get_neuron_id_dict(h5_path)`.
- `print_h5_contents`: Arguments:
- `visititems`: Function `visititems(group, func)`.
- `visitor_func`: prints name followed by a meangingful description of an hdf5 node.
- `check_dataset_format`: Checks whether dataset name is formatted properly
- `assert_h5_file`: Function `assert_h5_file(h5_path)`.
- `check_h5_file`: Check whether h5_path indeed points to h5
- `write_to_h5`: Writes data at data_path to .h5 file at h5_path
- `write_to_group`: Write data to hdf5 group
- `write_to_dataset`: write_to_groupwrite_to_group
- `read_h5`: Returns data at datapath from h5 file at h5_path
- `get_stim_chan`: Function `get_stim_chan(ons, min_th)`.
- `assert_recompute`: Function `assert_recompute(key, neuron_group, overwrite_h5, selective_overwrite)`.
- `assert_recompute_any`: Function `assert_recompute_any(keys, neuron_group, overwrite_h5, selective_overwrite)`.
- `h5_group_keys`: Returns list of keys of h5 file group
- `all_keys_in_group`: Function `all_keys_in_group(keys, group)`.
- `relative_unit_path_h5`: Function `relative_unit_path_h5(dataset, unit)`.

## `info.py`

- `sync_wr_chance_shadmehr`: y1 and y2 should be T trials x B bins 2D matrices
- `covariance`: y1 and y2 should be T trials x B bins 2D matrices
- `compute_sync_matrix`: Compute the P(A int B)/P(A)P(B) for each neuron A and B,
- `l2_synchrony`: Function `l2_synchrony(signal)`.
- `avg_synchrony`: Function `avg_synchrony(signal)`.
- `total_synchrony`: Signal: B x T x N tensor
- `total_var_synchrony`: Signal: B x T x N tensor
- `mgf_synchrony`: Signal: B x T x N matrix
- `lagged_synchrony_analysis`: Function `lagged_synchrony_analysis(signal, target, lags, res)`.
- `more_than_n_neurons_active`: Function `more_than_n_neurons_active(signal, res)`.
- `lagged_correlations`: Function `lagged_correlations(signal, target, lags, axis)`.
- `lagged_correlation`: Function `lagged_correlation(signal, target, lag, axis)`.
- `correlation`: Function `correlation(x, y, axis)`.
- `multivariate_mutual_information`: X is a Bins x Trials x Neurons matrix
- `total_correlation`: Generalization to N variables of mutual information (for 2 variables, same thing).
- `mutual_information`: KL divergence from the joint distribution P(X1,...,Xn)
- `multivariate_copula`: Compute the proba Q such that P(X1, ..., XN) ~= Q(X1, ..., XN)P(X1)...P(XN).
- `compute_p_joint`: returns the product proba of the configuration: p_joint(config),
- `compute_p_prod`: returns the proba of the configuration: p_product(config),
- `compute_p_prod2`: returns the proba of the configuration: p_joint(config),
- `kullback_leibler`: Compute the KL divergence of two probability measures, given by arrays of the same size that sum to one.
- `entropy`: Function `entropy(p, axis)`.
- `array_of_all_binaries`: returns an array with all binaries in {0, 1}^N,
- `broadcastable_shape`: Function `broadcastable_shape(m, n)`.
- `int_to_binary`: A dimension is added to a tensor of integers, such that the last dimension gives the binary decomposition,
- `equivalence_measure`: Give the probability that the two measures on {0,1}^N are equal,
- `cut_log`: returns log p if p > 0, else 0.
- `residual_cv2`: Returns:
- `Paintb_PaPb`: y1 and y2 should be T x B matrices (T trials and B bins)

## `inout.py`

- `read_metadata`: Function `read_metadata(dp)`.
- `metadata`: Read spikeGLX (.ap/lf.meta) or openEphys (.oebin) metadata files
- `chan_map`: Returns probe channel map.
- `predefined_chanmap`: Returns predefined channel map.
- `get_binary_file_path`: Function `get_binary_file_path(dp, filt_suffix, absolute_path)`.
- `get_meta_file_path`: Function `get_meta_file_path(dp, filt_suffix, absolute_path)`.
- `get_glx_file_path`: Return the path of a spikeGLX file (.bin or .meta) from a directory.
- `unpackbits`: unpacks numbers in bits.
- `get_npix_sync`: Unpacks neuropixels external input data, to align spikes to events.
- `extract_rawChunk`: Function to extract a chunk of raw data on a given range of channels on a given time window.
- `extract_binary_channel_subset`: Extract subset of channels from binary file into another binary file.
- `read_custom_binary`: Returns a memory map of a neuropixels binary file with a custom number of channels.
- `assert_chan_in_dataset`: Function `assert_chan_in_dataset(dp, channels, ignore_ks_chanfilt)`.
- `detect_hardware_filter`: Detects if the Neuropixels hardware filter was on or off during recording.
- `preprocess_binary_file`: Creates a preprocessed copy of binary file at dp/fname_filtered.bin,
- `make_preprocessing_fname`: Function `make_preprocessing_fname(fname, ADC_realign, median_subtract, f_low, f_high, filter_forward...)`.
- `detected_preprocessed_fname`: Function `detected_preprocessed_fname(fname)`.
- `paq_read`: Read PAQ file (from PackIO) into python

## `merger.py`

- `merge_datasets`: Merges datasets together and aligns data accross probes, modelling drift as a affine function.
- `ask_syncchan`: Function `ask_syncchan(ons)`.
- `get_ds_table`: Function `get_ds_table(dp)`.
- `get_dataset_id`: Arguments:
- `assert_same_dataset`: Asserts if all provided units belong to the same dataset.
- `assert_multi`: Function `assert_multi(dp)`.
- `get_ds_ids`: Function `get_ds_ids(U)`.
- `get_dataset_ids`: Arguments:
- `get_source_dp_u`: If dp is a merged datapath, returns datapath from source dataset and unit as integer.
- `merge_units_across_ss`: Merge units across ecephys spike sortings.

## `metrics.py`

- `quality_metrics`: Wrapper of calculate_quality_metrics to easily run on a kilosort directory.
- `calculate_quality_metrics`: Calculate metrics for all units on one probe
- `calculate_isi_violations`: Function `calculate_isi_violations(spike_times, spike_clusters, total_units, isi_threshold, min_isi)`.
- `calculate_presence_ratio`: Function `calculate_presence_ratio(spike_times, spike_clusters, total_units)`.
- `calculate_firing_rate`: Function `calculate_firing_rate(spike_times, spike_clusters, total_units)`.
- `calculate_amplitude_cutoff`: Function `calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units)`.
- `calculate_pc_metrics_one_cluster`: Function `calculate_pc_metrics_one_cluster(cluster_peak_channels, idx, cluster_id, cluster_ids, half_spread, pc_features...)`.
- `calculate_pc_metrics`: :param spike_clusters:
- `calculate_silhouette_score`: Function `calculate_silhouette_score(spike_clusters, spike_templates, total_units, pc_features, pc_feature_ind, total_spikes...)`.
- `calculate_drift_metrics`: Function `calculate_drift_metrics(spike_times, spike_clusters, spike_templates, total_units, pc_features, pc_feature_ind...)`.
- `isi_violations`: Calculate ISI violations for a spike train.
- `presence_ratio`: Calculate fraction of time the unit is present within an epoch.
- `firing_rate`: Calculate firing rate for a spike train.
- `amplitude_cutoff`: Calculate approximate fraction of spikes missing from a distribution of amplitudes
- `mahalanobis_metrics`: Calculates isolation distance and L-ratio (metrics computed from Mahalanobis distance)
- `lda_metrics`: Calculates d-prime based on Linear Discriminant Analysis
- `nearest_neighbors_metrics`: Calculates unit contamination based on NearestNeighbors search in PCA space
- `features_intersect`: # Take only the channels that have calculated features out of the ones we are interested in:
- `get_unit_pcs`: Return PC features for one unit
- `get_spike_depths`: Calculates the distance (in microns) of individual spikes from the probe tip

## `ml.py`

- `set_seed`: Function that controls randomness. NumPy and random modules must be imported.
- `run_cross_validation`: It runs a sklearn model with the best parameters found in the hyperparameter tuning step, and
- `umap_cached`: A simple way to save UMAP results when running it many times on the same data.
- `get_cluster_colors`: Function `get_cluster_colors(n_clusters, alpha, alpha_outliers)`.
- `labels_to_rgb_colors`: order: unique list of labels, to choose order of colors from colormap
- `red_dim_plot`: Function `red_dim_plot(X, dims_to_plot, labels, title, xlim, ylim...)`.

## `model.py`

- `generate_design_matrix`: Arguments:

## `preprocess.py`

- `whitening`: Whitens along axis 0.
- `whitening_matrix`: Compute the whitening matrix using ZCA.
- `load_ks_whitening_matrix`: Return kilosort whitening matrix
- `approximated_whitening_matrix`: Rather than computing the true whitening matrix from a signal x,
- `cov_to_whitening_matrix`: Function `cov_to_whitening_matrix(cov, nRange)`.
- `zca_whitening`: Function `zca_whitening(cov)`.
- `zca_whitening_local`: Function `zca_whitening_local(cov, nRange)`.
- `whitening_matrix_cpu`: wmat = whitening_matrix(dat, fudge=1e-18)
- `whiten_multimethod`: Whitens the input matrix X using specified whitening method.
- `med_substract`: Median substract along axis 0
- `bandpass_filter`: Butterworth bandpass filter.
- `apply_filter`: Apply a filter to an array, bidirectionally.
- `gpufilter`: Function `gpufilter(buff, fs, fslow, fshigh, order, car...)`.
- `get_filter_params`: Function `get_filter_params(fs, fshigh, fslow, order)`.
- `make_kernel`: Compile a kernel and pass optional constant ararys.
- `get_lfilter_kernel`: Function `get_lfilter_kernel(N, isfortran, reverse)`.
- `lfilter`: Perform a linear filter along the first axis on a GPU array.
- `pad`: Function `pad(fcn_convolve)`.
- `convolve_cpu`: CPU convolution based on scipy.signal.
- `convolve_gpu_direct`: Straight GPU FFT-based convolution that fits in memory.
- `convolve_gpu_chunked`: Chunked GPU FFT-based convolution for large arrays.
- `convolve_gpu`: Function `convolve_gpu(x, b)`.
- `svdecon`: Input:
- `svdecon_cpu`: Function `svdecon_cpu(X)`.
- `free_gpu_memory`: Function `free_gpu_memory()`.
- `cu_mean`: Function `cu_mean(x, axis)`.
- `cu_median`: Compute the median of a CuPy array on the GPU.
- `cu_var`: Function `cu_var(x)`.
- `cu_ones`: Function `cu_ones(shape, dtype, order)`.
- `cu_zscore`: Function `cu_zscore(a, axis)`.
- `adc_realign`: Function `adc_realign(data, version)`.
- `fshift`: Shifts a 1D or 2D signal in frequency domain, to allow for accurate non-integer shifts
- `adc_shifts`: The sampling is serial within the same ADC, but it happens at the same time in all ADCs.
- `kfilt`: Applies a butterworth filter on the 0-axis with tapering / padding
- `agc`: Automatic gain control
- `fcn_cosine`: Returns a soft thresholding function with a cosine taper:
- `ibl_convolve`: Frequency domain convolution along the last dimension (2d arrays)
- `ns_optim_fft`: Gets the next higher combination of factors of 2 and 3 than ns to compute efficient ffts

## `spk_t.py`

- `ids`: ********
- `load_amplitudes`: Function `load_amplitudes(dp, unit, verbose, periods, again, enforced_rp...)`.
- `trn`: Computes spike train (1, Nspikes) - int64, in samples
- `duplicates_mask`: - t: in samples, sampled at fs Hz
- `enforce_rp`: Enforce a refractory period of enforced_rp ms to a spike train.
- `isi`: ********
- `inst_cv2`: Arguments:
- `isint_filtered`: - t: array of time stamps, in samples
- `mean_firing_rate`: - t: array of time stamps, in samples
- `mean_inst_firing_rate`: - t: array of time stamps, in samples
- `coefficient_of_variation`: - t: array of time stamps, in samples
- `mfr`: Computes the mean firing rate of a unit.
- `binarize`: Function to turn a spike train (array of time stamps)
- `trnb`: ********
- `get_firing_periods`: Arguments:
- `firing_periods`: Arguments:
- `inst_firing_rate`: - again: bool, whether to recompute results rather than loading them from cache.
- `find_stable_recording_period`: Finds a locally optimal recording period (in terms of stability i.e. low std) of at least 'target_period' seconds.
- `train_quality`: Subselect spike times which meet two criteria:
- `trn_filtered`: Returns spike times (in sample) meeting the false positive and false negative criteria.
- `good_sections_from_mask`: Returns a list of good sections [[t1,t2], ...] in units of 'time_series'
- `get_common_good_sections`: Returns the intersection of sections (typically, [t1, t2] time windows) across N lists of sections.
- `gaussian_cut`: Function `gaussian_cut(x, a, mu, sigma, x_cut)`.
- `curve_fit_`: Function `curve_fit_(x, num, p1)`.
- `ampli_fit_gaussian_cut`: Function `ampli_fit_gaussian_cut(x, n_bins)`.
- `gaussian_amp_est`: Function `gaussian_amp_est(x, n_bins)`.
- `estimate_bins`: Function `estimate_bins(x, rule)`.
- `Freedman_Diaconis_bin_estimate`: Function `Freedman_Diaconis_bin_estimate(x)`.

## `spk_wvf.py`

- `wvf`: ********
- `get_waveforms`: Function `get_waveforms(dp, u, n_waveforms, t_waveforms, selection, periods...)`.
- `wvf_dsmatch`: ********
- `shift_match`: Iterate through waveforms to align them to each other
- `across_channels_SNR`: Function `across_channels_SNR(dp, u, n_waveforms, t_waveforms, periods, spike_ids...)`.
- `get_pc`: Function `get_pc(waveforms)`.
- `get_peak_chan`: Returns index of peak channel, either according to the full probe channel map (0 through 383)
- `get_depthSort_peakChans`: Usage:
- `get_chan_pos`: Returns (x, y) position of channel on Neuropixels probe.
- `get_peak_pos`: Returns [x,y] relative position on the probe in um (y=0 at probe tip).
- `get_chDis`: dp: datapath to dataset
- `templates`: ********
- `get_ids_subset`: Function `get_ids_subset(dp, unit, n_waveforms, batch_size_waveforms, selection, periods...)`.
- `select_waveforms_in_batch`: Batch selection of spikes.
- `data_chunk`: Get a data chunk.
- `excerpts`: Yield (start, end) where start is included and end is excluded.

## `stats.py`

- `pdf_normal`: Normal probability density function.
- `pdf_poisson`: Poisson probability density function.
- `cdf`: Function `cdf(X, _pdf, w1, b, *args)`.
- `cdf_normal`: Normal probability cumulative function.
- `cdf_poisson`: Poisson probability cumulative function.
- `fractile`: Function `fractile(p, _cdf, w1, w2, b, *args)`.
- `fractile_normal`: Fractile of order p drawn from the normal cumulative probability density function.
- `fractile_poisson`: Same for Poisson distribution.
- `check_outliers`: x: 1D numpy array
- `check_normality`: ASSUMPTIONS of 4 tests: Observations in each sample are independent and identically distributed (iid).
- `check_eqVariances`: Levene test - to do notably before ANOVA.
- `corrTest_pearson`: ->> TEST WHETHER TWO SAMPLES HAVE A LINEAR RELATIONSHIP
- `get_all_up_to_median`: Function `get_all_up_to_median(arr, window_a, window_b, hbin)`.
- `get_half_centered_on_mode`: Splits a distribution in two parts with equal AUC, centered on the mode of the distribution.
- `split_distr_N`: Splits a distribution in N parts with equal bin window (and optionally equal AUC).
- `get_isolated_stamps`: Returns elements of t1 surrounded by 2 intervals >= isolation_halfwin
- `get_synced_stamps`: Returns array t1, time stamps of two time X synchronous with Y.
- `get_CIH`: WARNING direction matters - will return CIH of 1 to 2, not 2 to 1.

## `utils.py`

- `docstring_decorator`: Feed as many arguments as wished to incorporate into the function f's docstring.
- `assert_float`: Function `assert_float(x)`.
- `assert_int`: Function `assert_int(x)`.
- `assert_iterable`: Function `assert_iterable(x)`.
- `npa`: Returns np.array of some kind.
- `save_np_array`: Save a numpy array to a file.
- `isnumeric`: Function `isnumeric(x)`.
- `sign`: Returns the sign of the input number (1 or -1). 1 for 0 or -0.
- `minus_is_1`: Function `minus_is_1(x)`.
- `read_pyfile`: Reads .py file and returns contents as dictionnary.
- `list_files`: List files with extension "extension" in directory "directory".
- `has_write_permission`: Check if the given path is writable without creating it.
- `has_space_left`: Check if there's enough space left at the given path.
- `is_writable`: Function `is_writable(path, required_space_mb)`.
- `pprint_dic`: Function `pprint_dic(dic)`.
- `repr_string`: Function `repr_string(self)`.
- `any_n_consec`: The trick below finds whether there are n_consec consecutive ones in the array comp
- `thresh_numba`: Returns indices of the data points closest to a directed crossing of th.
- `thresh`: Returns indices of the data points closest to a directed crossing of th.
- `thresh_fast`: Returns indices of the data points closest to a directed crossing of th.
- `thresh_consecutive`: Finds consecutive threshold crossings in a 1D array.
- `thresh_consec`: Returns indices and values of threshold crosses lasting >=n_consec consecutive samples in arr.
- `get_timestamps_in_windows_sorted`: Returns timestamps in windows defined by P.
- `get_timestamps_in_windows`: Returns timestamps in windows defined by P.
- `get_timestamps_in_windows_mask`: Function `get_timestamps_in_windows_mask(T, P)`.
- `zscore`: Returns z-scored (centered, reduced) array using outer edges of array to compute mean and std.
- `smooth`: Smoothes a 1D array or a 2D array along specified axis.
- `rolling_average`: Performs rolling average on x i.e. convolves x with a uniform window of length w.
- `slice_along_axis`: Returns properly formatted slice to slice array/list along specified axis.
- `xcorr_axis`: Cross-correlation between two Nd arrays
- `xcorr_1d_fft`: Cross-correlation along specific axis between two Nd arrays.
- `xcorr_1d_loop`: Cross-correlation along axis 0 between two 2D arrays.
- `xcorr_2d`: Cross-correlation along ALL axis between two Nd arrays.
- `normalize`: Vanilla normalization (center, reduce) along specified axis
- `get_bins`: Function `get_bins(cwin, cbin)`.
- `find_nearest`: Function `find_nearest(array, value)`.
- `mask_2d`: Mask a 2D array and preserve the
- `make_2D_array`: Function to get 2D array from a list of lists
- `split`: Arguments:
- `align_timeseries`: Usage 1: align >=2 time series in the same temporal reference frame with the same sampling frequency fs
- `align_timeseries_interpol`: Align a list of N timeseries in the temporal reference frame of the first timeserie.
- `peakdetect_parabole`: Misspelling of peakdetect_parabola
- `peakdetect`: Converted from/based on a MATLAB script at:
- `peakdetect_fft`: Performs a FFT calculation on the data and zero-pads the results to
- `peakdetect_parabola`: Function for detecting local maxima and minima in a signal.
- `peakdetect_sine`: Function for detecting local maxima and minima in a signal.
- `peakdetect_sine_locked`: Convenience function for calling the 'peakdetect_sine' function with
- `peakdetect_spline`: Performs a b-spline interpolation on the data to increase resolution and
- `peakdetect_zero_crossing`: Function for detecting local maxima and minima in a signal.
- `zero_crossings`: Algorithm to find zero crossings. Smooths the curve and finds the
- `zero_crossings_sine_fit`: Detects the zero crossings of a signal by fitting a sine model function


Total listed functions: **529**

