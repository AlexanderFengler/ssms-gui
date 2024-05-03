import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ssms.config import model_config
from ssms.basic_simulators.simulator import simulator
from matplotlib.lines import Line2D


def plot_func_model(
    model_name,
    theta,
    axis,
    value_range=None,
    n_samples=10,
    bin_size=0.05,
    add_data_rts=True,
    add_data_model_keep_slope=True,
    add_data_model_keep_boundary=True,
    add_data_model_keep_ndt=True,
    add_data_model_keep_starting_point=True,
    add_data_model_markersize_starting_point=50,
    add_data_model_markertype_starting_point=0,
    add_data_model_markershift_starting_point=0,
    n_trajectories = 0,
    linewidth_histogram=0.5,
    linewidth_model=0.5,
    legend_fontsize=12,
    legend_shadow=True,
    legend_location="upper right",
    data_color="blue",
    posterior_uncertainty_color="black",
    alpha=0.05,
    delta_t_model=0.001,
    random_state=None,
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

    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError("value_range keyword argument must be supplied.")

    if len(value_range) > 2:
        value_range = (value_range[0], value_range[-1])

    # Extract some parameters from kwargs
    bins = np.arange(value_range[0], value_range[-1], bin_size)

    if model_config[model_name]["nchoices"] > 2:
        raise ValueError("The model plot works only for 2 choice models at the moment")

    # RUN SIMULATIONS
    # -------------------------------

    # Simulator Data
    if random_state is not None:
        np.random.seed(random_state)
    
    rand_int = np.random.choice(400000000)
    sim_out = simulator(model = model_name, theta = theta, n_samples = n_samples, 
              no_noise = False, delta_t = 0.001, 
              bin_dim = None, random_state = rand_int)
    
    sim_out_traj = {}
    for i in range(n_trajectories):
        rand_int = np.random.choice(400000000)
        sim_out_traj[i] = simulator(model = model_name, theta = theta, n_samples = 1, 
                                    no_noise = False, delta_t = 0.001, 
                                    bin_dim = None, random_state = rand_int, smooth_unif = False)

    sim_out_no_noise = simulator(model = model_name, theta = theta, n_samples = 1, 
                                 no_noise = True, delta_t = 0.001, 
                                 bin_dim = None, smooth_unif = False)

    # ADD DATA HISTOGRAMS
    weights_up = np.tile(
        (1 / bin_size) / sim_out['rts'][(sim_out['rts'] != -999)].shape[0],
        reps=sim_out['rts'][(sim_out['rts'] != -999) & (sim_out['choices'] == 1)].shape[0],
    )
    weights_down = np.tile(
        (1 / bin_size) / sim_out['rts'][(sim_out['rts'] != -999)].shape[0],
        reps=sim_out['rts'][(sim_out['rts'] != -999) & (sim_out['choices'] != 1)].shape[0],
    )

    (b_high, b_low) = (np.maximum(sim_out['metadata']['boundary'], 0), 
                       np.minimum((-1) * sim_out['metadata']['boundary'], 0))

    # ADD HISTOGRAMS
    # -------------------------------

    ylim = kwargs.pop("ylim", 3)
    #hist_bottom = kwargs.pop("hist_bottom", 2)
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
        hist_bottom_high = b_high[0] #hist_bottom
        hist_bottom_low = -b_low[0] #hist_bottom

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

    axis_twin_up.hist(
        np.abs(sim_out['rts'][(sim_out['rts'] != -999) & (sim_out['choices'] == 1)]),
        bins=bins,
        weights=weights_up,
        histtype=hist_histtype,
        bottom=hist_bottom_high,
        alpha=1,
        color=data_color,
        edgecolor=data_color,
        linewidth=linewidth_histogram,
        zorder=-1,
    )

    axis_twin_down.hist(
        np.abs(sim_out['rts'][(sim_out['rts'] != -999) & (sim_out['choices'] != 1)]),
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

    # ADD MODEL:
    j = 0
    t_s = np.arange(0, sim_out['metadata']['max_t'], delta_t_model) #value_range[-1], delta_t_model)

    _add_model_cartoon_to_ax(
        sample=sim_out_no_noise,
        axis=axis,
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
        ylim_low=ylim_low,
        ylim_high=ylim_high,
        t_s=t_s,
        color=posterior_uncertainty_color,
        zorder_cnt=j,
    )

    if n_trajectories > 0:
        _add_trajectories(
            axis=axis,
            sample=sim_out_traj,
            t_s=t_s,
            delta_t_graph=delta_t_model,
            n_trajectories=n_trajectories,
            **kwargs,
        )
    
    return axis

# AF-TODO: Add documentation for this function
def _add_trajectories(
    axis=None,
    sample=None,
    t_s=None,
    delta_t_graph=0.01,
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
    """Add trajectories to a given axis."""
    # Check markercolor type
    if isinstance(markercolor_trajectory_rt_choice, str):
        markercolor_trajectory_rt_choice_dict = {}
        for value_ in sample[0]['metadata']['possible_choices']:
            markercolor_trajectory_rt_choice_dict[
                value_
            ] = markercolor_trajectory_rt_choice
    elif isinstance(markercolor_trajectory_rt_choice, list):
        cnt = 0
        for value_ in sample[0]['metadata']['possible_choices']:
            markercolor_trajectory_rt_choice_dict[
                value_
            ] = markercolor_trajectory_rt_choice[cnt]
            cnt += 1
    elif isinstance(markercolor_trajectory_rt_choice, dict):
        markercolor_trajectory_rt_choice_dict = markercolor_trajectory_rt_choice
    else:
        pass

    # Check trajectory color type
    if isinstance(color_trajectories, str):
        color_trajectories_dict = {}
        for value_ in sample[0]['metadata']['possible_choices']:
            color_trajectories_dict[value_] = color_trajectories
    elif isinstance(color_trajectories, list):
        cnt = 0
        for value_ in sample[0]['metadata']['possible_choices']:
            color_trajectories_dict[value_] = color_trajectories[cnt]
            cnt += 1
    elif isinstance(color_trajectories, dict):
        color_trajectories_dict = color_trajectories
    else:
        pass

    # Make bounds
    (b_high, b_low) = (np.maximum(sample[0]['metadata']['boundary'], 0), 
                       np.minimum((-1) * sample[0]['metadata']['boundary'], 0))
    
    b_h_init = b_high[0]
    b_l_init = b_low[0]
    n_roll = int((sample[0]['metadata']['t'][0] / delta_t_graph) + 1)
    b_high = np.roll(b_high, n_roll)
    b_high[:n_roll] = b_h_init
    b_low = np.roll(b_low, n_roll)
    b_low[:n_roll] = b_l_init

    print(n_trajectories)
    # Trajectories
    for i in range(n_trajectories):
        tmp_traj = sample[i]['metadata']['trajectory']
        tmp_traj_choice = float(sample[i]['choices'].flatten())
        maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)), t_s.shape[0])

        # Identify boundary value at timepoint of crossing
        b_tmp = b_high[maxid + n_roll] if tmp_traj_choice > 0 else b_low[maxid + n_roll]

        axis.plot(
            t_s[:maxid] + sample[i]['metadata']['t'][0], #sample.t.values[0],
            tmp_traj[:maxid],
            color=color_trajectories_dict[tmp_traj_choice],
            alpha=alpha_trajectories,
            linewidth=linewidth_trajectories,
            zorder=2000 + i,
        )

        if highlight_trajectory_rt_choice:
            axis.scatter(
                t_s[maxid] + sample[i]['metadata']['t'][0], #sample.t.values[0],
                b_tmp,
                # tmp_traj[maxid],
                markersize_trajectory_rt_choice,
                color=markercolor_trajectory_rt_choice_dict[tmp_traj_choice],
                alpha=1,
                marker=markertype_trajectory_rt_choice,
                zorder=2000 + i,
            )

# AF-TODO: Add documentation to this function
def _add_model_cartoon_to_ax(
    sample=None,
    axis=None,
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
    (b_high, b_low) = (np.maximum(sample['metadata']['boundary'], 0), 
                       np.minimum((-1) * sample['metadata']['boundary'], 0))

    b_h_init = b_high[0]
    b_l_init = b_low[0]
    n_roll = int((sample['metadata']['t'][0] / delta_t_graph) + 1)
    b_high = np.roll(b_high, n_roll)
    b_high[:n_roll] = b_h_init
    b_low = np.roll(b_low, n_roll)
    b_low[:n_roll] = b_l_init

    tmp_traj = sample["metadata"]["trajectory"]
    maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)),
                       t_s.shape[0])

    if keep_boundary:
        # Upper bound
        axis.plot(
            t_s,  # + sample.t.values[0],
            b_high[:t_s.shape[0]],
            color=color,
            alpha=1,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
            label=tmp_label,
        )

        # Lower bound
        axis.plot(
            t_s,  # + sample.t.values[0],
            b_low[:t_s.shape[0]],
            color=color,
            alpha=1,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
        )

    # Slope
    if keep_slope:
        axis.plot(
            t_s[:maxid] + sample['metadata']['t'][0],
            tmp_traj[:maxid],
            color=color,
            alpha=1,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
        )  # TOOK AWAY LABEL

    # Non-decision time
    if keep_ndt:
        axis.axvline(
            x=sample['metadata']['t'][0],
            ymin=ylim_low,
            ymax=ylim_high,
            color=color,
            linestyle="--",
            linewidth=lw_m,
            zorder=1000 + zorder_cnt,
            alpha=1,
        )
    # Starting point
    if keep_starting_point:
        axis.scatter(
            sample['metadata']['t'][0] + markershift_starting_point,
            b_low[0] + (sample['metadata']['z'][0] * (b_high[0] - b_low[0])),
            markersize_starting_point,
            marker=markertype_starting_point,
            color=color,
            alpha=1,
            zorder=1000 + zorder_cnt,
        )

def plot_func_model_n(
    model_name,
    theta,
    axis,
    n_trajectories=10,
    value_range=None,
    bin_size=0.05,
    n_samples=10,
    linewidth_histogram=0.5,
    linewidth_model=0.5,
    legend_fontsize=7,
    legend_shadow=True,
    legend_location="upper right",
    delta_t_model=0.001,
    add_legend=True,
    alpha=1.0,
    keep_frame=False,
    random_state=None,
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
    # ------------
    ylim = kwargs.pop("ylim", 4)

    axis.set_xlim(value_range[0], value_range[-1])
    axis.set_ylim(0, ylim)

    # ADD MODEL:

    # RUN SIMULATIONS
    # -------------------------------

    # Simulator Data
    if random_state is not None:
        np.random.seed(random_state)
    
    rand_int = np.random.choice(400000000)
    sim_out = simulator(model = model_name, theta = theta, n_samples = n_samples, 
              no_noise = False, delta_t = 0.001, 
              bin_dim = None, random_state = rand_int)
    
    choices = sim_out['metadata']['possible_choices']
    
    sim_out_traj = {}
    for i in range(n_trajectories):
        rand_int = np.random.choice(400000000)
        sim_out_traj[i] = simulator(model = model_name, theta = theta, n_samples = 1, 
                                    no_noise = False, delta_t = 0.001, 
                                    bin_dim = None, random_state = rand_int, smooth_unif = False)

    sim_out_no_noise = simulator(model = model_name, theta = theta, n_samples = 1, 
                                 no_noise = True, delta_t = 0.001, 
                                 bin_dim = None, smooth_unif = False)

    # ADD HISTOGRAMS
    # -------------------------------

    # POSTERIOR BASED HISTOGRAM
    j = 0
    b = np.maximum(sim_out['metadata']['boundary'], 0)
    bottom = b[0]
    for choice in choices:
        tmp_label = None

        if add_legend and j == 0:
            tmp_label = "PostPred"

        weights = np.tile(
            (1 / bin_size) / sim_out['rts'].shape[0],
            reps=sim_out['rts'][(sim_out['choices'] == choice) & (sim_out['rts'] != -999)].shape[0],
        )

        axis.hist(
            np.abs(sim_out['rts'][(sim_out['choices'] == choice) & (sim_out['rts'] != -999)]),
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

    # ADD MODEL:
    tmp_label = None
    j = 0
    t_s = np.arange(0, sim_out['metadata']['max_t'], delta_t_model)

    if add_legend and (j == 0):
        tmp_label = "PostPred"

    _add_model_n_cartoon_to_ax(
        sample=sim_out_no_noise,
        axis=axis,
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

    if n_trajectories > 0:
        _add_trajectories_n(
            axis=axis,
            sample=sim_out_traj,
            t_s=t_s,
            delta_t_graph=delta_t_model,
            n_trajectories=n_trajectories,
            **kwargs,
        )

    if add_legend:
        custom_elems = [
            Line2D([0], [0], color=color_dict[choice], lw=1) for choice in choices
        ]
        custom_titles = ["response: " + str(choice) for choice in choices]

        custom_elems.append(
            Line2D([0], [0], color="black", lw=1.0, linestyle="dashed")
        )
        # custom_elems.append(Line2D([0], [0], color="black", lw=1.0, linestyle="-"))
        # custom_titles.append("Data")
        # custom_titles.append("Posterior")

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

    return axis

def _add_trajectories_n(axis=None,
                        sample=None,
                        t_s=None,
                        delta_t_graph=0.01,
                        n_trajectories=10,
                        highlight_trajectory_rt_choice=True,
                        markersize_trajectory_rt_choice=50,
                        markertype_trajectory_rt_choice="*",
                        markercolor_trajectory_rt_choice="black",
                        linewidth_trajectories=1,
                        alpha_trajectories=0.5,
                        color_trajectories="black",
                        **kwargs,
                        ):
    
    """Add trajectories to a given axis."""
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

    # Check trajectory color type
    if isinstance(color_trajectories, str):
        color_trajectories_dict = {}
        for value_ in sample[0]['metadata']['possible_choices']:
            color_trajectories_dict[value_] = color_trajectories
    elif isinstance(color_trajectories, list):
        cnt = 0
        for value_ in sample[0]['metadata']['possible_choices']:
            color_trajectories_dict[value_] = color_trajectories[cnt]
            cnt += 1
    elif isinstance(color_trajectories, dict):
        color_trajectories_dict = color_trajectories
    else:
        pass

    # Make bounds
    b = np.maximum(sample[0]['metadata']['boundary'], 0)
    b_init = b[0]
    n_roll = int((sample[0]['metadata']['t'][0] / delta_t_graph) + 1)
    b = np.roll(b, n_roll)
    b[:n_roll] = b_init

    # Trajectories
    for i in range(n_trajectories):
        tmp_traj = sample[i]['metadata']['trajectory']
        tmp_traj_choice = float(sample[i]['choices'].flatten())

        for j in range(len(sample[i]['metadata']['possible_choices'])):
            tmp_maxid = np.minimum(np.argmax(np.where(tmp_traj[:, j] > -999)), t_s.shape[0])

            # Identify boundary value at timepoint of crossing
            b_tmp = b[tmp_maxid + n_roll]

            axis.plot(
                t_s[:tmp_maxid] + sample[i]['metadata']['t'][0], #sample.t.values[0],
                tmp_traj[:tmp_maxid, j],
                color=color_dict[j],
                alpha=alpha_trajectories,
                linewidth=linewidth_trajectories,
                zorder=2000 + i,
            )

            if highlight_trajectory_rt_choice and tmp_traj_choice == j:
                axis.scatter(
                    t_s[tmp_maxid] + sample[i]['metadata']['t'][0], #sample.t.values[0],
                    b_tmp,
                    # tmp_traj[maxid],
                    markersize_trajectory_rt_choice,
                    color=color_dict[tmp_traj_choice],
                    alpha=1,
                    marker=markertype_trajectory_rt_choice,
                    zorder=2000 + i,
                )
            elif highlight_trajectory_rt_choice and tmp_traj_choice != j:
                axis.scatter(
                    t_s[tmp_maxid] + sample[i]['metadata']['t'][0] + 0.05, #sample.t.values[0],
                    tmp_traj[tmp_maxid, j],
                    # tmp_traj[maxid],
                    markersize_trajectory_rt_choice,
                    color=color_dict[j],
                    alpha=1,
                    marker=5,
                    zorder=2000 + i,
                )

def _add_model_n_cartoon_to_ax(
    sample=None,
    axis=None,
    delta_t_graph=None,
    sample_hist_alpha=None,
    keep_boundary=True,
    keep_ndt=True,
    keep_slope=True,
    keep_starting_point=True,
    lw_m=None,
    linestyle="-",
    tmp_label=None,
    ylim=None,
    t_s=None,
    zorder_cnt=1,
    color_dict=None,
):
    
    b = np.maximum(sample['metadata']['boundary'], 0)
    b_init = b[0]
    n_roll = int((sample['metadata']['t'][0] / delta_t_graph) + 1)
    b = np.roll(b, n_roll)
    b[:n_roll] = b_init

    # Upper bound
    if keep_boundary:
        axis.plot(
            t_s,
            b[:t_s.shape[0]],
            color="black",
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
            linestyle=linestyle,
            label=tmp_label,
        )

    # Starting point
    if keep_starting_point:
        axis.axvline(
            x=sample['metadata']['t'][0],
            ymin=-ylim,
            ymax=ylim,
            color="black",
            linestyle=linestyle,
            linewidth=lw_m,
            alpha=sample_hist_alpha,
        )

    # # MAKE SLOPES (VIA TRAJECTORIES HERE --> RUN NOISE FREE SIMULATIONS)!
    if keep_slope:
        tmp_traj = sample["metadata"]["trajectory"]

        for i in range(len(sample["metadata"]["possible_choices"])):
            tmp_maxid = np.minimum(np.argmax(np.where(tmp_traj[:, i] > -999)), t_s.shape[0])

            # Slope
            axis.plot(
                t_s[:tmp_maxid] + sample['metadata']['t'][0],
                tmp_traj[:tmp_maxid, i],
                color=color_dict[i],
                linestyle=linestyle,
                alpha=sample_hist_alpha,
                zorder=1000 + zorder_cnt,
                linewidth=lw_m,
            )  # TOOK AWAY LABEL

    return b[0]