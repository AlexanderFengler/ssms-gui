def _plot_func_model(
    bottom_node,
    axis,
    value_range=None,
    samples=10,
    bin_size=0.05,
    add_data_rts=True,
    add_data_model=True,
    add_data_model_keep_slope=True,
    add_data_model_keep_boundary=True,
    add_data_model_keep_ndt=True,
    add_data_model_keep_starting_point=True,
    add_data_model_markersize_starting_point=50,
    add_data_model_markertype_starting_point=0,
    add_data_model_markershift_starting_point=0,
    add_posterior_uncertainty_model=False,
    add_posterior_uncertainty_rts=False,
    add_posterior_mean_model=True,
    add_posterior_mean_rts=True,
    add_trajectories=False,
    data_label="Data",
    secondary_data=None,
    secondary_data_label=None,
    secondary_data_color="blue",
    linewidth_histogram=0.5,
    linewidth_model=0.5,
    legend_fontsize=12,
    legend_shadow=True,
    legend_location="upper right",
    data_color="blue",
    posterior_mean_color="red",
    posterior_uncertainty_color="black",
    alpha=0.05,
    delta_t_model=0.01,
    add_legend=True,  # keep_frame=False,
    **kwargs,
):
    """Calculate posterior predictive for a certain bottom node.

    Arguments:
        bottom_node: pymc.stochastic
            Bottom node to compute posterior over.

        axis: matplotlib.axis
            Axis to plot into.

        value_range: numpy.ndarray
            Range over which to evaluate the likelihood.

    Optional:
        samples: int <default=10>
            Number of posterior samples to use.

        bin_size: float <default=0.05>
            Size of bins used for histograms

        alpha: float <default=0.05>
            alpha (transparency) level for the sample-wise elements of the plot

        add_data_rts: bool <default=True>
            Add data histogram of rts ?

        add_data_model: bool <default=True>
            Add model cartoon for data

        add_posterior_uncertainty_rts: bool <default=True>
            Add sample by sample histograms?

        add_posterior_mean_rts: bool <default=True>
            Add a mean posterior?

        add_model: bool <default=True>
            Whether to add model cartoons to the plot.

        linewidth_histogram: float <default=0.5>
            linewdith of histrogram plot elements.

        linewidth_model: float <default=0.5>
            linewidth of plot elements concerning the model cartoons.

        legend_location: str <default='upper right'>
            string defining legend position. Find the rest of the options in the matplotlib documentation.

        legend_shadow: bool <default=True>
            Add shadow to legend box?

        legend_fontsize: float <default=12>
            Fontsize of legend.

        data_color : str <default="blue">
            Color for the data part of the plot.

        posterior_mean_color : str <default="red">
            Color for the posterior mean part of the plot.

        posterior_uncertainty_color : str <default="black">
            Color for the posterior uncertainty part of the plot.

        delta_t_model:
            specifies plotting intervals for model cartoon elements of the graphs.
    """

    # AF-TODO: Add a mean version of this!
    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError("value_range keyword argument must be supplied.")

    if len(value_range) > 2:
        value_range = (value_range[0], value_range[-1])

    # Extract some parameters from kwargs
    bins = np.arange(value_range[0], value_range[-1], bin_size)

    # If bottom_node is a DataFrame we know that we are just plotting real data
    if type(bottom_node) == pd.DataFrame:
        samples_tmp = [bottom_node]
        data_tmp = None
    else:
        samples_tmp = _post_pred_generate(
            bottom_node,
            samples=samples,
            data=None,
            append_data=False,
            add_model_parameters=True,
        )
        data_tmp = bottom_node.value.copy()

    # Relevant for recovery mode
    node_data_full = kwargs.pop("node_data", None)

    tmp_model = kwargs.pop("model_", "angle")
    if len(model_config[tmp_model]["choices"]) > 2:
        raise ValueError("The model plot works only for 2 choice models at the moment")

    # ---------------------------

    ylim = kwargs.pop("ylim", 3)
    hist_bottom = kwargs.pop("hist_bottom", 2)
    hist_histtype = kwargs.pop("hist_histtype", "step")

    if ("ylim_high" in kwargs) and ("ylim_low" in kwargs):
        ylim_high = kwargs["ylim_high"]
        ylim_low = kwargs["ylim_low"]
    else:
        ylim_high = ylim
        ylim_low = -ylim

    if ("hist_bottom_high" in kwargs) and ("hist_bottom_low" in kwargs):
        hist_bottom_high = kwargs["hist_bottom_high"]
        hist_bottom_low = kwargs["hist_bottom_low"]
    else:
        hist_bottom_high = hist_bottom
        hist_bottom_low = hist_bottom

    axis.set_xlim(value_range[0], value_range[-1])
    axis.set_ylim(ylim_low, ylim_high)
    axis_twin_up = axis.twinx()
    axis_twin_down = axis.twinx()
    axis_twin_up.set_ylim(ylim_low, ylim_high)
    axis_twin_up.set_yticks([])
    axis_twin_down.set_ylim(ylim_high, ylim_low)
    axis_twin_down.set_yticks([])
    axis_twin_down.set_axis_off()
    axis_twin_up.set_axis_off()

    # ADD HISTOGRAMS
    # -------------------------------
    # POSTERIOR BASED HISTOGRAM
    if add_posterior_uncertainty_rts:  # add_uc_rts:
        j = 0
        for sample in samples_tmp:
            tmp_label = None

            if add_legend and j == 0:
                tmp_label = "PostPred"

            weights_up = np.tile(
                (1 / bin_size) / sample.shape[0],
                reps=sample.loc[sample.response == 1, :].shape[0],
            )
            weights_down = np.tile(
                (1 / bin_size) / sample.shape[0],
                reps=sample.loc[(sample.response != 1), :].shape[0],
            )

            axis_twin_up.hist(
                np.abs(sample.rt[sample.response == 1]),
                bins=bins,
                weights=weights_up,
                histtype=hist_histtype,
                bottom=hist_bottom_high,
                alpha=alpha,
                color=posterior_uncertainty_color,
                edgecolor=posterior_uncertainty_color,
                zorder=-1,
                label=tmp_label,
                linewidth=linewidth_histogram,
            )

            axis_twin_down.hist(
                np.abs(sample.loc[(sample.response != 1), :].rt),
                bins=bins,
                weights=weights_down,
                histtype=hist_histtype,
                bottom=hist_bottom_low,
                alpha=alpha,
                color=posterior_uncertainty_color,
                edgecolor=posterior_uncertainty_color,
                linewidth=linewidth_histogram,
                zorder=-1,
            )
            j += 1

    if add_posterior_mean_rts:  # add_mean_rts:
        concat_data = pd.concat(samples_tmp)
        tmp_label = None

        if add_legend:
            tmp_label = "PostPred Mean"

        weights_up = np.tile(
            (1 / bin_size) / concat_data.shape[0],
            reps=concat_data.loc[concat_data.response == 1, :].shape[0],
        )
        weights_down = np.tile(
            (1 / bin_size) / concat_data.shape[0],
            reps=concat_data.loc[(concat_data.response != 1), :].shape[0],
        )

        axis_twin_up.hist(
            np.abs(concat_data.rt[concat_data.response == 1]),
            bins=bins,
            weights=weights_up,
            histtype=hist_histtype,
            bottom=hist_bottom_high,
            alpha=1.0,
            color=posterior_mean_color,
            edgecolor=posterior_mean_color,
            zorder=-1,
            label=tmp_label,
            linewidth=linewidth_histogram,
        )

        axis_twin_down.hist(
            np.abs(concat_data.loc[(concat_data.response != 1), :].rt),
            bins=bins,
            weights=weights_down,
            histtype=hist_histtype,
            bottom=hist_bottom_low,
            alpha=1.0,
            color=posterior_mean_color,
            edgecolor=posterior_mean_color,
            linewidth=linewidth_histogram,
            zorder=-1,
        )

    # DATA HISTOGRAM
    if (data_tmp is not None) and add_data_rts:
        tmp_label = None
        if add_legend:
            tmp_label = data_label

        weights_up = np.tile(
            (1 / bin_size) / data_tmp.shape[0],
            reps=data_tmp[data_tmp.response == 1].shape[0],
        )
        weights_down = np.tile(
            (1 / bin_size) / data_tmp.shape[0],
            reps=data_tmp[(data_tmp.response != 1)].shape[0],
        )

        axis_twin_up.hist(
            np.abs(data_tmp[data_tmp.response == 1].rt),
            bins=bins,
            weights=weights_up,
            histtype=hist_histtype,
            bottom=hist_bottom_high,
            alpha=1,
            color=data_color,
            edgecolor=data_color,
            label=tmp_label,
            zorder=-1,
            linewidth=linewidth_histogram,
        )

        axis_twin_down.hist(
            np.abs(data_tmp[(data_tmp.response != 1)].rt),
            bins=bins,
            weights=weights_down,
            histtype=hist_histtype,
            bottom=hist_bottom_low,
            alpha=1,
            color=data_color,
            edgecolor=data_color,
            linewidth=linewidth_histogram,
            zorder=-1,
        )

    # SECONDARY DATA HISTOGRAM
    if secondary_data is not None:
        tmp_label = None
        if add_legend:
            if secondary_data_label is not None:
                tmp_label = secondary_data_label

        weights_up = np.tile(
            (1 / bin_size) / secondary_data.shape[0],
            reps=secondary_data[secondary_data.response == 1].shape[0],
        )
        weights_down = np.tile(
            (1 / bin_size) / secondary_data.shape[0],
            reps=secondary_data[(secondary_data.response != 1)].shape[0],
        )

        axis_twin_up.hist(
            np.abs(secondary_data[secondary_data.response == 1].rt),
            bins=bins,
            weights=weights_up,
            histtype=hist_histtype,
            bottom=hist_bottom_high,
            alpha=1,
            color=secondary_data_color,
            edgecolor=secondary_data_color,
            label=tmp_label,
            zorder=-100,
            linewidth=linewidth_histogram,
        )

        axis_twin_down.hist(
            np.abs(secondary_data[(secondary_data.response != 1)].rt),
            bins=bins,
            weights=weights_down,
            histtype=hist_histtype,
            bottom=hist_bottom_low,
            alpha=1,
            color=secondary_data_color,
            edgecolor=secondary_data_color,
            linewidth=linewidth_histogram,
            zorder=-100,
        )
    # -------------------------------

    if add_legend:
        if data_tmp is not None:
            axis_twin_up.legend(
                fontsize=legend_fontsize, shadow=legend_shadow, loc=legend_location
            )

    # ADD MODEL:
    j = 0
    t_s = np.arange(0, value_range[-1], delta_t_model)

    # MAKE BOUNDS (FROM MODEL CONFIG) !
    if add_posterior_uncertainty_model:  # add_uc_model:
        for sample in samples_tmp:
            _add_model_cartoon_to_ax(
                sample=sample,
                axis=axis,
                tmp_model=tmp_model,
                keep_slope=add_data_model_keep_slope,
                keep_boundary=add_data_model_keep_boundary,
                keep_ndt=add_data_model_keep_ndt,
                keep_starting_point=add_data_model_keep_starting_point,
                markersize_starting_point=add_data_model_markersize_starting_point,
                markertype_starting_point=add_data_model_markertype_starting_point,
                markershift_starting_point=add_data_model_markershift_starting_point,
                delta_t_graph=delta_t_model,
                sample_hist_alpha=alpha,
                lw_m=linewidth_model,
                tmp_label=tmp_label,
                ylim_low=ylim_low,
                ylim_high=ylim_high,
                t_s=t_s,
                color=posterior_uncertainty_color,
                zorder_cnt=j,
            )

    if (node_data_full is not None) and add_data_model:
        _add_model_cartoon_to_ax(
            sample=node_data_full,
            axis=axis,
            tmp_model=tmp_model,
            keep_slope=add_data_model_keep_slope,
            keep_boundary=add_data_model_keep_boundary,
            keep_ndt=add_data_model_keep_ndt,
            keep_starting_point=add_data_model_keep_starting_point,
            markersize_starting_point=add_data_model_markersize_starting_point,
            markertype_starting_point=add_data_model_markertype_starting_point,
            markershift_starting_point=add_data_model_markershift_starting_point,
            delta_t_graph=delta_t_model,
            sample_hist_alpha=1.0,
            lw_m=linewidth_model + 0.5,
            tmp_label=None,
            ylim_low=ylim_low,
            ylim_high=ylim_high,
            t_s=t_s,
            color=data_color,
            zorder_cnt=j + 1,
        )

    if add_posterior_mean_model:  # add_mean_model:
        tmp_label = None
        if add_legend:
            tmp_label = "PostPred Mean"

        _add_model_cartoon_to_ax(
            sample=pd.DataFrame(pd.concat(samples_tmp).mean().astype(np.float32)).T,
            axis=axis,
            tmp_model=tmp_model,
            keep_slope=add_data_model_keep_slope,
            keep_boundary=add_data_model_keep_boundary,
            keep_ndt=add_data_model_keep_ndt,
            keep_starting_point=add_data_model_keep_starting_point,
            markersize_starting_point=add_data_model_markersize_starting_point,
            markertype_starting_point=add_data_model_markertype_starting_point,
            markershift_starting_point=add_data_model_markershift_starting_point,
            delta_t_graph=delta_t_model,
            sample_hist_alpha=1.0,
            lw_m=linewidth_model + 0.5,
            tmp_label=None,
            ylim_low=ylim_low,
            ylim_high=ylim_high,
            t_s=t_s,
            color=posterior_mean_color,
            zorder_cnt=j + 2,
        )

    if add_trajectories:
        _add_trajectories(
            axis=axis,
            sample=samples_tmp[0],
            tmp_model=tmp_model,
            t_s=t_s,
            delta_t_graph=delta_t_model,
            **kwargs,
        )


# AF-TODO: Add documentation for this function
def _add_trajectories(
    axis=None,
    sample=None,
    t_s=None,
    delta_t_graph=0.01,
    tmp_model=None,
    n_trajectories=10,
    supplied_trajectory=None,
    maxid_supplied_trajectory=1,  # useful for gifs
    highlight_trajectory_rt_choice=True,
    markersize_trajectory_rt_choice=50,
    markertype_trajectory_rt_choice="*",
    markercolor_trajectory_rt_choice="red",
    linewidth_trajectories=1,
    alpha_trajectories=0.5,
    color_trajectories="black",
    **kwargs,
):
    # Check markercolor type
    if type(markercolor_trajectory_rt_choice) == str:
        markercolor_trajectory_rt_choice_dict = {}
        for value_ in model_config[tmp_model]["choices"]:
            markercolor_trajectory_rt_choice_dict[
                value_
            ] = markercolor_trajectory_rt_choice
    elif type(markercolor_trajectory_rt_choice) == list:
        cnt = 0
        for value_ in model_config[tmp_model]["choices"]:
            markercolor_trajectory_rt_choice_dict[
                value_
            ] = markercolor_trajectory_rt_choice[cnt]
            cnt += 1
    elif type(markercolor_trajectory_rt_choice) == dict:
        markercolor_trajectory_rt_choice_dict = markercolor_trajectory_rt_choice
    else:
        pass

    # Check trajectory color type
    if type(color_trajectories) == str:
        color_trajectories_dict = {}
        for value_ in model_config[tmp_model]["choices"]:
            color_trajectories_dict[value_] = color_trajectories
    elif type(color_trajectories) == list:
        cnt = 0
        for value_ in model_config[tmp_model]["choices"]:
            color_trajectories_dict[value_] = color_trajectories[cnt]
            cnt += 1
    elif type(color_trajectories) == dict:
        color_trajectories_dict = color_trajectories
    else:
        pass

    # Make bounds
    (b_low, b_high) = _make_bounds(
        tmp_model=tmp_model,
        sample=sample,
        delta_t_graph=delta_t_graph,
        t_s=t_s,
        return_shifted_by_ndt=False,
    )

    # Trajectories
    if supplied_trajectory is None:
        for i in range(n_trajectories):
            rand_int = np.random.choice(400000000)
            out_traj = simulator(
                theta=sample[model_config[tmp_model]["params"]].values[0],
                model=tmp_model,
                n_samples=1,
                no_noise=False,
                delta_t=delta_t_graph,
                bin_dim=None,
                random_state=rand_int,
            )

            tmp_traj = out_traj[2]["trajectory"]
            tmp_traj_choice = float(out_traj[1].flatten())
            maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)), t_s.shape[0])

            # Identify boundary value at timepoint of crossing
            b_tmp = b_high[maxid] if tmp_traj_choice > 0 else b_low[maxid]

            axis.plot(
                t_s[:maxid] + sample.t.values[0],
                tmp_traj[:maxid],
                color=color_trajectories_dict[tmp_traj_choice],
                alpha=alpha_trajectories,
                linewidth=linewidth_trajectories,
                zorder=2000 + i,
            )

            if highlight_trajectory_rt_choice:
                axis.scatter(
                    t_s[maxid] + sample.t.values[0],
                    b_tmp,
                    # tmp_traj[maxid],
                    markersize_trajectory_rt_choice,
                    color=markercolor_trajectory_rt_choice_dict[tmp_traj_choice],
                    alpha=1,
                    marker=markertype_trajectory_rt_choice,
                    zorder=2000 + i,
                )

    else:
        if len(supplied_trajectory["trajectories"].shape) == 1:
            supplied_trajectory["trajectories"] = np.expand_dims(
                supplied_trajectory["trajectories"], axis=0
            )

        for j in range(supplied_trajectory["trajectories"].shape[0]):
            maxid = np.minimum(
                np.argmax(np.where(supplied_trajectory["trajectories"][j, :] > -999)),
                t_s.shape[0],
            )
            if j == (supplied_trajectory["trajectories"].shape[0] - 1):
                maxid_traj = min(maxid, maxid_supplied_trajectory)
            else:
                maxid_traj = maxid

            axis.plot(
                t_s[:maxid_traj] + sample.t.values[0],
                supplied_trajectory["trajectories"][j, :maxid_traj],
                color=color_trajectories_dict[
                    supplied_trajectory["trajectory_choices"][j]
                ],  # color_trajectories,
                alpha=alpha_trajectories,
                linewidth=linewidth_trajectories,
                zorder=2000 + j,
            )

            # Identify boundary value at timepoint of crossing
            b_tmp = (
                b_high[maxid_traj]
                if supplied_trajectory["trajectory_choices"][j] > 0
                else b_low[maxid_traj]
            )

            if maxid_traj == maxid:
                if highlight_trajectory_rt_choice:
                    axis.scatter(
                        t_s[maxid_traj] + sample.t.values[0],
                        b_tmp,
                        # supplied_trajectory['trajectories'][j, maxid_traj],
                        markersize_trajectory_rt_choice,
                        color=markercolor_trajectory_rt_choice_dict[
                            supplied_trajectory["trajectory_choices"][j]
                        ],  # markercolor_trajectory_rt_choice,
                        alpha=1,
                        marker=markertype_trajectory_rt_choice,
                        zorder=2000 + j,
                    )


# AF-TODO: Add documentation to this function
def _add_model_cartoon_to_ax(
    sample=None,
    axis=None,
    tmp_model=None,
    keep_slope=True,
    keep_boundary=True,
    keep_ndt=True,
    keep_starting_point=True,
    markersize_starting_point=80,
    markertype_starting_point=1,
    markershift_starting_point=-0.05,
    delta_t_graph=None,
    sample_hist_alpha=None,
    lw_m=None,
    tmp_label=None,
    ylim_low=None,
    ylim_high=None,
    t_s=None,
    zorder_cnt=1,
    color="black",
):
    # Make bounds
    b_low, b_high = _make_bounds(
        tmp_model=tmp_model,
        sample=sample,
        delta_t_graph=delta_t_graph,
        t_s=t_s,
        return_shifted_by_ndt=True,
    )

    # MAKE SLOPES (VIA TRAJECTORIES HERE --> RUN NOISE FREE SIMULATIONS)!
    out = simulator(
        theta=sample[model_config[tmp_model]["params"]].values[0],
        model=tmp_model,
        n_samples=1,
        no_noise=True,
        delta_t=delta_t_graph,
        bin_dim=None,
    )

    tmp_traj = out[2]["trajectory"]
    maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)), t_s.shape[0])

    if "hddm_base" in tmp_model:
        a_tmp = sample.a.values[0] / 2
        tmp_traj = tmp_traj - a_tmp

    if keep_boundary:
        # Upper bound
        axis.plot(
            t_s,  # + sample.t.values[0],
            b_high,
            color=color,
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
            label=tmp_label,
        )

        # Lower bound
        axis.plot(
            t_s,  # + sample.t.values[0],
            b_low,
            color=color,
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
        )

    # Slope
    if keep_slope:
        axis.plot(
            t_s[:maxid] + sample.t.values[0],
            tmp_traj[:maxid],
            color=color,
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
        )  # TOOK AWAY LABEL

    # Non-decision time
    if keep_ndt:
        axis.axvline(
            x=sample.t.values[0],
            ymin=ylim_low,
            ymax=ylim_high,
            color=color,
            linestyle="--",
            linewidth=lw_m,
            zorder=1000 + zorder_cnt,
            alpha=sample_hist_alpha,
        )
    # Starting point
    if keep_starting_point:
        axis.scatter(
            sample.t.values[0] + markershift_starting_point,
            b_low[0] + (sample.z.values[0] * (b_high[0] - b_low[0])),
            markersize_starting_point,
            marker=markertype_starting_point,
            color=color,
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
        )


def _make_bounds(
    tmp_model=None,
    sample=None,
    delta_t_graph=None,
    t_s=None,
    return_shifted_by_ndt=True,
):
    # MULTIPLICATIVE BOUND
    if tmp_model == "weibull" or tmp_model == "weibull_cdf":
        b = np.maximum(
            sample.a.values[0]
            * model_config[tmp_model]["boundary"](
                t=t_s, alpha=sample.alpha.values[0], beta=sample.beta.values[0]
            ),
            0,
        )

        # Move boundary forward by the non-decision time
        b_raw_high = deepcopy(b)
        b_raw_low = deepcopy(-b)
        b_init_val = b[0]
        t_shift = np.arange(0, sample.t.values[0], delta_t_graph).shape[0]
        b = np.roll(b, t_shift)
        b[:t_shift] = b_init_val

    # ADDITIVE BOUND
    elif tmp_model == "angle":
        b = np.maximum(
            sample.a.values[0]
            + model_config[tmp_model]["boundary"](t=t_s, theta=sample.theta.values[0]),
            0,
        )

        b_raw_high = deepcopy(b)
        b_raw_low = deepcopy(-b)
        # Move boundary forward by the non-decision time
        b_init_val = b[0]
        t_shift = np.arange(0, sample.t.values[0], delta_t_graph).shape[0]
        b = np.roll(b, t_shift)
        b[:t_shift] = b_init_val

    # CONSTANT BOUND
    elif (
        tmp_model == "ddm"
        or tmp_model == "ornstein"
        or tmp_model == "levy"
        or tmp_model == "full_ddm"
        or tmp_model == "ddm_hddm_base"
        or tmp_model == "full_ddm_hddm_base"
    ):
        b = sample.a.values[0] * np.ones(t_s.shape[0])

        if "hddm_base" in tmp_model:
            b = (sample.a.values[0] / 2) * np.ones(t_s.shape[0])

        b_raw_high = b
        b_raw_low = -b

    # Separate out upper and lower bound:
    b_high = b
    b_low = -b

    if return_shifted_by_ndt:
        return (b_low, b_high)
    else:
        return (b_raw_low, b_raw_high)


def _plot_func_model_n(
    bottom_node,
    axis,
    value_range=None,
    samples=10,
    bin_size=0.05,
    add_posterior_uncertainty_model=False,
    add_posterior_uncertainty_rts=False,
    add_posterior_mean_model=True,
    add_posterior_mean_rts=True,
    linewidth_histogram=0.5,
    linewidth_model=0.5,
    legend_fontsize=7,
    legend_shadow=True,
    legend_location="upper right",
    delta_t_model=0.01,
    add_legend=True,
    alpha=0.01,
    keep_frame=False,
    **kwargs,
):
    """Calculate posterior predictive for a certain bottom node.

    Arguments:
        bottom_node: pymc.stochastic
            Bottom node to compute posterior over.

        axis: matplotlib.axis
            Axis to plot into.

        value_range: numpy.ndarray
            Range over which to evaluate the likelihood.

    Optional:
        samples: int <default=10>
            Number of posterior samples to use.

        bin_size: float <default=0.05>
            Size of bins used for histograms

        alpha: float <default=0.05>
            alpha (transparency) level for the sample-wise elements of the plot

        add_posterior_uncertainty_rts: bool <default=True>
            Add sample by sample histograms?

        add_posterior_mean_rts: bool <default=True>
            Add a mean posterior?

        add_model: bool <default=True>
            Whether to add model cartoons to the plot.

        linewidth_histogram: float <default=0.5>
            linewdith of histrogram plot elements.

        linewidth_model: float <default=0.5>
            linewidth of plot elements concerning the model cartoons.

        legend_loc: str <default='upper right'>
            string defining legend position. Find the rest of the options in the matplotlib documentation.

        legend_shadow: bool <default=True>
            Add shadow to legend box?

        legend_fontsize: float <default=12>
            Fontsize of legend.

        data_color : str <default="blue">
            Color for the data part of the plot.

        posterior_mean_color : str <default="red">
            Color for the posterior mean part of the plot.

        posterior_uncertainty_color : str <default="black">
            Color for the posterior uncertainty part of the plot.


        delta_t_model:
            specifies plotting intervals for model cartoon elements of the graphs.
    """

    color_dict = {
        -1: "black",
        0: "black",
        1: "green",
        2: "blue",
        3: "red",
        4: "orange",
        5: "purple",
        6: "brown",
    }

    # AF-TODO: Add a mean version of this !
    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError("value_range keyword argument must be supplied.")

    if len(value_range) > 2:
        value_range = (value_range[0], value_range[-1])

    # Extract some parameters from kwargs
    bins = np.arange(value_range[0], value_range[-1], bin_size)

    # Relevant for recovery mode
    node_data_full = kwargs.pop("node_data", None)
    tmp_model = kwargs.pop("model_", "angle")

    bottom = 0
    # ------------
    ylim = kwargs.pop("ylim", 3)

    choices = model_config[tmp_model]["choices"]

    # If bottom_node is a DataFrame we know that we are just plotting real data
    if type(bottom_node) == pd.DataFrame:
        samples_tmp = [bottom_node]
        data_tmp = None
    else:
        samples_tmp = _post_pred_generate(
            bottom_node,
            samples=samples,
            data=None,
            append_data=False,
            add_model_parameters=True,
        )
        data_tmp = bottom_node.value.copy()

    axis.set_xlim(value_range[0], value_range[-1])
    axis.set_ylim(0, ylim)

    # ADD MODEL:
    j = 0
    t_s = np.arange(0, value_range[-1], delta_t_model)

    # # MAKE BOUNDS (FROM MODEL CONFIG) !
    if add_posterior_uncertainty_model:  # add_uc_model:
        for sample in samples_tmp:
            tmp_label = None

            if add_legend and (j == 0):
                tmp_label = "PostPred"

            _add_model_n_cartoon_to_ax(
                sample=sample,
                axis=axis,
                tmp_model=tmp_model,
                delta_t_graph=delta_t_model,
                sample_hist_alpha=alpha,
                lw_m=linewidth_model,
                tmp_label=tmp_label,
                linestyle="-",
                ylim=ylim,
                t_s=t_s,
                color_dict=color_dict,
                zorder_cnt=j,
            )

            j += 1

    if add_posterior_mean_model:  # add_mean_model:
        tmp_label = None
        if add_legend:
            tmp_label = "PostPred Mean"

        bottom = _add_model_n_cartoon_to_ax(
            sample=pd.DataFrame(pd.concat(samples_tmp).mean().astype(np.float32)).T,
            axis=axis,
            tmp_model=tmp_model,
            delta_t_graph=delta_t_model,
            sample_hist_alpha=1.0,
            lw_m=linewidth_model + 0.5,
            linestyle="-",
            tmp_label=None,
            ylim=ylim,
            t_s=t_s,
            color_dict=color_dict,
            zorder_cnt=j + 2,
        )

    if node_data_full is not None:
        _add_model_n_cartoon_to_ax(
            sample=node_data_full,
            axis=axis,
            tmp_model=tmp_model,
            delta_t_graph=delta_t_model,
            sample_hist_alpha=1.0,
            lw_m=linewidth_model + 0.5,
            linestyle="dashed",
            tmp_label=None,
            ylim=ylim,
            t_s=t_s,
            color_dict=color_dict,
            zorder_cnt=j + 1,
        )

    # ADD HISTOGRAMS
    # -------------------------------

    # POSTERIOR BASED HISTOGRAM
    if add_posterior_uncertainty_rts:  # add_uc_rts:
        j = 0
        for sample in samples_tmp:
            for choice in choices:
                tmp_label = None

                if add_legend and j == 0:
                    tmp_label = "PostPred"

                weights = np.tile(
                    (1 / bin_size) / sample.shape[0],
                    reps=sample.loc[sample.response == choice, :].shape[0],
                )

                axis.hist(
                    np.abs(sample.rt[sample.response == choice]),
                    bins=bins,
                    bottom=bottom,
                    weights=weights,
                    histtype="step",
                    alpha=alpha,
                    color=color_dict[choice],
                    zorder=-1,
                    label=tmp_label,
                    linewidth=linewidth_histogram,
                )
                j += 1

    if add_posterior_mean_rts:
        concat_data = pd.concat(samples_tmp)
        for choice in choices:
            tmp_label = None
            if add_legend and (choice == choices[0]):
                tmp_label = "PostPred Mean"

            weights = np.tile(
                (1 / bin_size) / concat_data.shape[0],
                reps=concat_data.loc[concat_data.response == choice, :].shape[0],
            )

            axis.hist(
                np.abs(concat_data.rt[concat_data.response == choice]),
                bins=bins,
                bottom=bottom,
                weights=weights,
                histtype="step",
                alpha=1.0,
                color=color_dict[choice],
                zorder=-1,
                label=tmp_label,
                linewidth=linewidth_histogram,
            )

    # DATA HISTOGRAM
    if data_tmp is not None:
        for choice in choices:
            tmp_label = None
            if add_legend and (choice == choices[0]):
                tmp_label = "Data"

            weights = np.tile(
                (1 / bin_size) / data_tmp.shape[0],
                reps=data_tmp.loc[data_tmp.response == choice, :].shape[0],
            )

            axis.hist(
                np.abs(data_tmp.rt[data_tmp.response == choice]),
                bins=bins,
                bottom=bottom,
                weights=weights,
                histtype="step",
                linestyle="dashed",
                alpha=1.0,
                color=color_dict[choice],
                edgecolor=color_dict[choice],
                zorder=-1,
                label=tmp_label,
                linewidth=linewidth_histogram,
            )
    # -------------------------------

    if add_legend:
        if data_tmp is not None:
            custom_elems = [
                Line2D([0], [0], color=color_dict[choice], lw=1) for choice in choices
            ]
            custom_titles = ["response: " + str(choice) for choice in choices]

            custom_elems.append(
                Line2D([0], [0], color="black", lw=1.0, linestyle="dashed")
            )
            custom_elems.append(Line2D([0], [0], color="black", lw=1.0, linestyle="-"))
            custom_titles.append("Data")
            custom_titles.append("Posterior")

            axis.legend(
                custom_elems,
                custom_titles,
                fontsize=legend_fontsize,
                shadow=legend_shadow,
                loc=legend_location,
            )

    # FRAME
    if not keep_frame:
        axis.set_frame_on(False)


def _add_model_n_cartoon_to_ax(
    sample=None,
    axis=None,
    tmp_model=None,
    delta_t_graph=None,
    sample_hist_alpha=None,
    lw_m=None,
    linestyle="-",
    tmp_label=None,
    ylim=None,
    t_s=None,
    zorder_cnt=1,
    color_dict=None,
):
    if "weibull" in tmp_model:
        b = np.maximum(
            sample.a.values[0]
            * model_config[tmp_model]["boundary"](
                t=t_s, alpha=sample.alpha.values[0], beta=sample.beta.values[0]
            ),
            0,
        )

    elif "angle" in tmp_model:
        b = np.maximum(
            sample.a.values[0]
            + model_config[tmp_model]["boundary"](t=t_s, theta=sample.theta.values[0]),
            0,
        )

    else:
        b = sample.a.values[0] * np.ones(t_s.shape[0])

    # Upper bound
    axis.plot(
        t_s + sample.t.values[0],
        b,
        color="black",
        alpha=sample_hist_alpha,
        zorder=1000 + zorder_cnt,
        linewidth=lw_m,
        linestyle=linestyle,
        label=tmp_label,
    )

    # Starting point
    axis.axvline(
        x=sample.t.values[0],
        ymin=-ylim,
        ymax=ylim,
        color="black",
        linestyle=linestyle,
        linewidth=lw_m,
        alpha=sample_hist_alpha,
    )

    # # MAKE SLOPES (VIA TRAJECTORIES HERE --> RUN NOISE FREE SIMULATIONS)!
    out = simulator(
        theta=sample[model_config[tmp_model]["params"]].values[0],
        model=tmp_model,
        n_samples=1,
        no_noise=True,
        delta_t=delta_t_graph,
        bin_dim=None,
    )

    # # AF-TODO: Add trajectories
    tmp_traj = out[2]["trajectory"]

    for i in range(len(model_config[tmp_model]["choices"])):
        tmp_maxid = np.minimum(np.argmax(np.where(tmp_traj[:, i] > -999)), t_s.shape[0])

        # Slope
        axis.plot(
            t_s[:tmp_maxid] + sample.t.values[0],
            tmp_traj[:tmp_maxid, i],
            color=color_dict[i],
            linestyle=linestyle,
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
        )  # TOOK AWAY LABEL

    return b[0]