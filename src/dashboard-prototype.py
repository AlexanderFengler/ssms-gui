import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

# from hssm import simulate_data
from ssms.config import model_config
from ssms.basic_simulators.simulator import simulator
import pandas as pd
import utils


# Function to create input select widgets
def create_param_selectors(model_name: str, model_num: int = 1):
    d_config = model_config[model_name]
    params = d_config["params"]
    param_bounds_low = d_config["param_bounds"][0]
    param_bounds_high = d_config["param_bounds"][1]
    param_defaults = d_config["default_params"]

    d_param_slider = {}
    for i, (name, low, high, default) in enumerate(
        zip(
            params,
            param_bounds_low,
            param_bounds_high,
            param_defaults,
        )
    ):
        d_param_slider[i] = st.slider(
            label=name,
            min_value=float(low),
            max_value=float(high),
            value=float(default),
            key=f"param{i}"
            f"_{model_name}"
            f"_{model_num}"
            f'_{st.session_state["slider_version"]}',
        )
    return d_param_slider


def add_model():
    pass


def reset_sliders():
    st.session_state["slider_version"] += 1


# Initialize a slider version attribute state. Is used for resetting values
if "slider_version" not in st.session_state:
    st.session_state["slider_version"] = 1


def draw_model_configurator(model_num=1):
    # Create widgets for the sidebar
    # st.markdown("<h2 style='text-align: center; color: black;'>Model Configurator</h1>",
    #             unsafe_allow_html=True)
    # Dropdown selection of model name
    model_select = st.selectbox(
        "Model " + str(model_num), l_model_names, key="modelname" + str(model_num)
    )
    # Sliders for param values
    d_slider = create_param_selectors(model_select, model_num=model_num)
    # Number of data points to simulate
    nsamples = st.number_input("NSamples", value=5000, key="size" + str(model_num))
    # Number of trajectories to show
    ntrajectories = st.number_input(
        "NTrajectories", value=0, key="ntraj" + str(model_num)
    )
    return model_select, d_slider, nsamples, ntrajectories


st.set_page_config(layout="wide")

# Get list of model names
l_model_names = list(model_config.keys())

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        model_select_1, d_slider_1, nsamples_1, ntrajectories_1 = (
            draw_model_configurator(model_num=1)
        )
    with col2:
        model_select_2, d_slider_2, nsamples_2, ntrajectories_2 = (
            draw_model_configurator(model_num=2)
        )

    # Button to reset sliders to default values
    randomseed = st.number_input("RandomSeed", value=0, key="seed")
    st.button(
        "Reset",
        help="Reset parameters to defaults",
        key="reset",
        on_click=reset_sliders,
    )

# st.title("SSM Model Plots", )
st.markdown(
    "<h1 style='text-align: center; color: black;'>SSM Model Plots</h1>",
    unsafe_allow_html=True,
)

# Display components for main panel
fig1, ax1 = plt.subplots()
if model_config[model_select_1]["nchoices"] == 2 and not ("race" in model_select_1):
    ax1 = utils.utils.plot_func_model(
        model_name=model_select_1,
        theta=[list(d_slider_1.values())],
        axis=ax1,
        value_range=[-0.1, 5],
        n_samples=nsamples_1,
        ylim=5,
        data_color="blue",
        add_trajectories=True,
        n_trajectories=ntrajectories_1,
        linewidth_model=1,
        linewidth_histogram=1,
        random_state=randomseed,
    )
else:
    ax1 = utils.utils.plot_func_model_n(
        model_name=model_select_1,
        theta=[list(d_slider_1.values())],
        axis=ax1,
        value_range=[-0.1, 5],
        n_samples=nsamples_1,
        data_color="blue",
        add_trajectories=True,
        n_trajectories=ntrajectories_1,
        linewidth_model=1,
        linewidth_histogram=1,
        random_state=randomseed,
    )
ax1.set_title("Model 1")
ax1.set_xlabel("RT in seconds")

fig2, ax2 = plt.subplots()
if model_config[model_select_2]["nchoices"] == 2 and not ("race" in model_select_2):
    ax2 = utils.utils.plot_func_model(
        model_name=model_select_2,
        theta=[list(d_slider_2.values())],
        axis=ax2,
        value_range=[-0.1, 5],
        n_samples=nsamples_2,
        ylim=5,
        data_color="red",
        add_trajectories=True,
        n_trajectories=ntrajectories_2,
        linewidth_model=1,
        linewidth_histogram=1,
        random_state=randomseed,
    )
else:
    ax2 = utils.utils.plot_func_model_n(
        model_name=model_select_2,
        theta=[list(d_slider_2.values())],
        axis=ax2,
        value_range=[-0.1, 5],
        n_samples=nsamples_2,
        data_color="red",
        add_trajectories=True,
        n_trajectories=ntrajectories_2,
        linewidth_model=1,
        linewidth_histogram=1,
        random_state=randomseed,
    )

ax2.set_title("Model 2")
ax2.set_xlabel("RT in seconds")

# Place figure in placeholder
col1, col2 = st.columns(2)
with col1:
    figure_placeholder_1 = st.empty()  # Placeholder for figure render
    figure_placeholder_1.pyplot(fig1)
with col2:
    figure_placeholder_2 = st.empty()  # Placeholder for figure render
    figure_placeholder_2.pyplot(fig2)

# Simulate two datasets:
sim_output_1 = simulator(
    model=model_select_1,
    theta=[list(d_slider_1.values())],
    n_samples=nsamples_1,
    random_state=randomseed,
)
sim_output_2 = simulator(
    model=model_select_2,
    theta=[list(d_slider_2.values())],
    n_samples=nsamples_2,
    random_state=randomseed,
)

# Make metadata dataframe
metadata = pd.DataFrame(
    {
        "Model 1": [
            sim_output_1["metadata"]["model"],
            sim_output_1["choice_p"][0, 0],
            sim_output_1["rts"].mean(),
            sim_output_1["metadata"]["s"],
        ],
        "Model 2": [
            sim_output_2["metadata"]["model"],
            sim_output_2["choice_p"][0, 0],
            sim_output_2["rts"].mean(),
            sim_output_2["metadata"]["s"],
        ],
    },
    index=["Model", "Choice Probability", "Mean RT", "Noise SD"],
)

col3, col4 = st.columns(2)
with col3:
    if (
        len(sim_output_1["metadata"]["possible_choices"])
        == 2 | len(sim_output_2["metadata"]["possible_choices"])
        == 2
    ):
        figure_placeholder_3 = st.empty()

        # Plot the simulated data
        fig3, ax3 = plt.subplots()
        ax3.hist(
            sim_output_1["rts"][np.abs(sim_output_1["rts"]) != 999]
            * sim_output_1["choices"][np.abs(sim_output_1["rts"] != 999)],
            histtype="step",
            bins=50,
            density=True,
            color="blue",
            fill=None,
            label="Model 1",
        )
        ax3.hist(
            sim_output_2["rts"][np.abs(sim_output_2["rts"]) != 999]
            * sim_output_2["choices"][np.abs(sim_output_2["rts"] != 999)],
            histtype="step",
            bins=50,
            density=True,
            color="red",
            fill=None,
            label="Model 2",
        )
        ax3.legend()
        ax3.set_xlabel("RT")
        ax3.set_xlim(-5, 5)
        figure_placeholder_3.pyplot(fig3)
    else:
        # TODO: Implement better comparison plot
        # for models with more than 2 choice options
        pass
with col4:
    st.dataframe(metadata)
