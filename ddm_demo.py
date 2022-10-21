import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", context="paper", palette="rocket")

st.set_page_config(
    page_title="DDM Demo",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "https://www.twitter.com/moltaire",
        "About": "# DDM Demo. I made this for teaching purposes. Maybe it's useful for you, too.",
    },
)
np.random.seed(123)


# Model parameters
T = 100  # number of points per trajectory
nbins = 20

with st.sidebar:

    st.title("Drift Diffusion Model")
    st.write("Interactive simulation of the standard Drift Diffusion Model (DDM).")

    st.markdown("---")
    st.markdown("### Basic model parameters:")

    ## Drift
    v = st.slider(
        label="Drift rate ν",
        min_value=-0.1,
        max_value=0.1,
        step=0.01,
        value=0.03,
        help=r"The drift rate parameter $v$ controls the drift rate. Positive values result in average movement towards the upper boundary, negative values towards the lower boundary. Higher absolute values are often interpreted indicating faster processing and ability, but these interpretations depend on the exact context of the task.",
    )

    ## Noise
    sigma = st.slider(
        label="Diffusion coefficient σ:",
        min_value=0.0,
        max_value=0.5,
        step=0.01,
        value=0.05,
        help=r"The diffusion parameter $\sigma$ controls the amount of random noise that is added to the process at each time point. Note, that conventionally, the diffusion coefficient is often held constant (e.g., at 0.1 or 1.0) to allow estimation of the drift rate and boundary separation.",
    )

    ## Starting point
    z = st.slider(
        label="Starting point ζ:",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.25,
        help=r"The starting point parameter $\zeta$ controls the initial position between the boundaries. It is parameterized so that a value of 0.5 is right in the middle between the boundaries.",
    )

    ## Boundary
    b = st.slider(
        label="Boundary separation β:",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        value=2.0,
        help=r"The boundary separation $\beta$ determines the distance between the two boundaries. Therefore it controls the amount of evidence required to elicit a response. Note, that one would typically hold constant either the boundary separation, the drift rate, or the diffusion coefficient, otherwise the model parameters are not identifiable.",
    )

    ## NDT
    tau = st.slider(
        label="Non-decision-time τ:",
        min_value=0,
        max_value=T,
        step=1,
        value=20,
        help=r"The non-decision-time $\tau$ is thought to capture non-decision-related components of the response time, like stimulus encoding and motor preparation and execution.",
    )

    with st.expander("Advanced model parameters"):

        ## Drift rate variability
        sv = st.slider(
            label="Drift rate variability sν",
            min_value=0.0,
            max_value=0.1,
            step=0.01,
            value=0.0,
            help=r"Assuming a Gaussian distribution of drift rates across trials, this parameter determines the standard deviation of the trial drift rates $v$.",
        )

        ## Starting point variability
        sz = st.slider(
            label="Starting point variability sζ",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=0.0,
            help=r"Assuming a uniform distribution of starting points across trials, this parameter determines the width of this uniform distribution around $\zeta$.",
        )

        ## NDT variability
        stau = st.slider(
            label="Non-decision-time variability sτ",
            min_value=0,
            max_value=20,
            step=1,
            value=0,
            help=r"Assuming a uniform distribution of non-decision-times across trials, this parameter determines the width of this uniform distribution around $\tau$.",
        )

    with st.expander("Other settings"):

        # Allow the user to choose one trajectory
        n = st.slider(
            label="Number of trajectories:",
            min_value=1,
            max_value=200,
            step=1,
            value=100,
            help="Set the number of trajectories to be plotted. More looks nicer and gives smoother histograms, but might be a little more taxing for your computer to deal with.",
        )
        # Plotting backend
        plotting_backend = st.radio("Plot with...", ("matplotlib", "plotly"))

# Simulate data
t = np.arange(T)
epsilon = np.random.normal(scale=sigma, size=(n, T))
vs = np.random.normal(loc=v, scale=sv, size=n)
zs = np.clip(
    np.random.uniform(low=z - sz / 2, high=z + sz / 2, size=n), a_min=0.01, a_max=0.99
)
taus = np.clip(
    np.random.uniform(low=tau - stau / 2, high=tau + stau / 2, size=n), a_min=0, a_max=T
).astype(int)

df = pd.DataFrame(
    {
        f"y{i}": b * (zs[i] - 1 / 2)
        + np.hstack(
            [np.zeros(taus[i] + 1), (vs[i] + epsilon[i, : (T - taus[i] - 1)]).cumsum()]
        )
        for i in range(n)
    }
)
# Remove all data points outside of boundaries
df[(df.abs() >= b / 2).astype(int).cumsum(axis=0) > 1] = np.nan
# Set all values over the bound to the bound
df[df >= b / 2] = b / 2
df[df <= -b / 2] = -b / 2

# Add time index
df["x"] = np.arange(T)
df_long = pd.wide_to_long(df, stubnames="y", i="x", j="i").reset_index()

# Make a DataFrame of responses and response times
y_final = df[[f"y{i}" for i in range(n)]].ffill().iloc[-1]
responses = np.where(y_final <= -b / 2, -1, np.where(y_final >= b / 2, 1, np.nan))
rts = T - df.isnull().sum(axis=0)
data = pd.DataFrame(dict(response=responses, rt=rts), index=y_final.index)

# Compute y-limit for histogram by rounding the count in the largest bin to the next 10
bins = np.linspace(0, T, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
counts_up, _ = np.histogram(data.loc[data["response"] == 1, "rt"], bins=bins)
counts_down, _ = np.histogram(data.loc[data["response"] == -1, "rt"], bins=bins)
hist_ylim = 10 * np.ceil(np.max(np.concatenate([counts_up, counts_down])) / 10)

# Generate figure

## Mean drift computations
y0 = b * (z - 1 / 2)
if v == 0:
    x1 = T
    y1 = y0
elif v > 0:
    x1 = tau + (b / 2 - y0) // v + 1
    y1 = b / 2
else:
    x1 = tau + (-b / 2 - y0) // v + 1
    y1 = -b / 2

## Histograms
if plotting_backend == "plotly":
    hist_up = px.bar(x=bin_centers, y=counts_up)
    hist_up.update_yaxes(range=[0, hist_ylim], title="Count")

    # hist_down = px.histogram(data.loc[data["response"] == -1], x="rt", nbins=10)
    hist_down = px.bar(x=bin_centers, y=counts_down)
    hist_down.update_yaxes(range=[hist_ylim, 0], title="Count")

    for hist in [hist_up, hist_down]:
        hist.update_layout(height=100, bargap=0.075)
        hist.update_xaxes(showticklabels=False, range=[0, T], title=None, showgrid=True)

    ## Plot selected trajectory
    trajectories = px.line(df_long, x="x", y="y", line_group="i")
    trajectories.update_traces(opacity=np.max([0.1, 1 / n]))

    ### Add choice boundaries
    for bound in [-b / 2, b / 2]:
        trajectories.add_shape(
            type="line",
            x0=0,
            y0=bound,
            x1=T,
            y1=bound,
            line=dict(color="Black"),
            xref="x",
            yref="y",
        )

    ### Add 0 line
    trajectories.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=T,
        y1=0,
        line=dict(color="Black", width=0.5),
        xref="x",
        yref="y",
    )

    ### Add mean drift
    trajectories.add_shape(
        type="line",
        x0=tau,
        y0=y0,
        x1=x1,
        y1=y1,
        line=dict(color="Blue", width=2),
        xref="x",
        yref="y",
    )

    trajectories.update_yaxes(range=[1.1 * -b / 2, 1.1 * b / 2], title="Evidence")
    trajectories.update_xaxes(title="Time", range=[0, T])

    # Reduce margins
    for fig in [hist_up, trajectories, hist_down]:
        fig.update_layout(margin=dict(l=20, r=20, t=0, b=0))

    st.plotly_chart(hist_up)
    st.plotly_chart(trajectories)
    st.plotly_chart(hist_down)

elif plotting_backend == "matplotlib":
    fig, axs = plt.subplots(3, 1, gridspec_kw={"height_ratios": [1, 3, 1]}, sharex=True)

    # Trajectories
    traj_alpha = np.max([0.075, 1 / n**0.5])
    axs[1].plot(
        df[[f"y{i}" for i in range(n)]].values,
        color="#FF4B4B",
        alpha=traj_alpha,
    )
    # Plot starting points
    axs[1].plot(
        taus, -b / 2 + zs * b, "o", markersize=4, color="#FF4B4B", alpha=traj_alpha
    )

    ## Mean drift
    axs[1].plot([tau, x1], [y0, y1], color="#FF4B4B", lw=2)
    if sv > 0:
        from scipy.stats import norm

        t = np.arange(tau, T)
        axs[1].fill_between(
            t,
            y0 + norm.ppf(0.95, loc=v, scale=sv) * (t - tau),
            y0 + norm.ppf(0.05, loc=v, scale=sv) * (t - tau),
            color="#FF4B4B",
            alpha=0.2,
        )

    ## 0-line
    axs[1].axhline(0, color="k", lw=0.5, zorder=-1)
    axs[1].plot([T - 7.25], [0], ">", markersize=4, alpha=0.7)
    axs[1].text(T, 0, "Time", ha="right", va="center", backgroundcolor="white")

    ## Limits and labels
    axs[1].set_ylim(-b / 2, b / 2)
    axs[1].set_ylabel("Evidence")
    axs[1].text(
        T, b / 2, "Upper bound", ha="right", va="center", backgroundcolor="white"
    )
    axs[1].text(
        T, -b / 2, "Lower bound", ha="right", va="center", backgroundcolor="white"
    )

    # Histograms
    axs[0].bar(bin_centers, counts_up, width=5, color="#FF4B4B")
    axs[2].bar(bin_centers, counts_down, width=5, color="#FF4B4B")

    ## Limits and labels
    axs[0].set_ylim(0, hist_ylim)
    axs[0].set_ylabel("Upper\nresponses", va="center", ha="center")
    axs[2].set_ylim(hist_ylim, 0)
    axs[2].set_ylabel("Lower\nresponses", va="center", ha="center")
    axs[2].set_xlim(0, T)
    axs[2].set_xticks([])

    # Figure styling
    sns.despine(ax=axs[0], left=False, top=True, right=True)
    sns.despine(ax=axs[1], left=False, top=False, right=True, bottom=False)
    sns.despine(ax=axs[2], left=False, top=False, right=True, bottom=True)
    fig.tight_layout()
    fig.align_ylabels(axs)

    st.pyplot(fig)

st.markdown(
    f"""---
Upper responses:  {(data['response'] == 1).sum()}       ({(data['response'] == 1).mean() * 100:.2f}%)  
Lower responses:  {(data['response'] == -1).sum()}      ({(data['response'] == -1).mean() * 100:.2f}%)  
No response:      {pd.isnull(data['response']).sum()}   ({pd.isnull(data['response']).mean() * 100:.2f}%)
"""
)

with st.expander("Inspect data:"):
    st.dataframe(data, height=500)
    st.dataframe(df, height=500)
    
