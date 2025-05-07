# %%
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import data.nqDataLoader as nq
from pathlib import Path
import kaleido

# %%
# Data Loading
DATA_PATH = Path("data")
ASSETS_PATH = Path("assets")

def load_ground_truth(dataset: str) -> pd.DataFrame:
    data_dir = DATA_PATH / dataset
    ground_truth = pd.read_csv(data_dir / f"GT_DataPD_{dataset}.csv")
    return (
        ground_truth.set_index("pID")
        .rename(
            columns={
                "gt": "has_PD",
                "updrs108": "UPDRS-III",
                "afTap": "alternating_finger_tapping",
                "sTap": "single_finger_tapping",
                "nqScore": "nQi",
                "typingSpeed": "typing_speed",
            }
        )
        .assign(
            dataset=dataset
        )
    )

def load_all_trials(ground_truth: pd.DataFrame) -> pd.DataFrame:
    trial_data = []
    for tup in ground_truth.itertuples():
        if str(tup.file_1) != 'nan':
            fp = DATA_PATH / tup.dataset / f"data_{tup.dataset}" / tup.file_1
            key_presses_1, hold_times_1, press_times_1, release_times_1 = nq.get_data_filt_helper(
                fp
            )
        else:
            key_presses_1 = None
            hold_times_1 = None
            press_times_1 = None
            release_times_1 = None
        if str(tup.file_2) != 'nan':
            fp = DATA_PATH / tup.dataset / f"data_{tup.dataset}" / tup.file_2
            key_presses_2, hold_times_2, press_times_2, release_times_2 = nq.get_data_filt_helper(
                fp
            )
        else:
            key_presses_2 = None
            hold_times_2 = None
            press_times_2 = None
            release_times_2 = None
        data = {
            "pID": tup.Index,
            "has_PD": tup.has_PD,
            "UPDRS-III": tup._2,
            "alternating_finger_tapping": tup.alternating_finger_tapping,
            "single_finger_tapping": tup.single_finger_tapping,
            "nQi": tup.nQi,
            "typing_speed": tup.typing_speed,
            "key_presses_1": key_presses_1,
            "hold_times_1": hold_times_1,
            "press_times_1": press_times_1,
            "release_times_1": release_times_1,
            "key_presses_2": key_presses_2,
            "hold_times_2": hold_times_2,
            "press_times_2": press_times_2,
            "release_times_2": release_times_2,
        }
        trial_data.append(data)
    return pd.DataFrame(trial_data)

ground_truth_1 = load_ground_truth("MIT-CS1PD")
ground_truth_2 = load_ground_truth("MIT-CS2PD")
ground_truth = pd.concat([ground_truth_1, ground_truth_2])

all_trials = load_all_trials(ground_truth)

# %% [markdown]
# # Graph 1 - Average Key Delay vs. Has PD
# %%
def calculate_key_delays(press_times: np.ndarray, release_times: np.ndarray) -> float:
    if press_times is None or release_times is None or len(press_times) < 2 or len(release_times) < 2:
        return np.nan
    return press_times[1:] - release_times[:-1]

def calculate_average_key_delays(df: pd.DataFrame) -> pd.DataFrame:
    df["key_delays_1"] = df.apply(
        lambda row: calculate_key_delays(row["press_times_1"], row["release_times_1"]),
        axis=1
    )
    df["key_delays_2"] = df.apply(
        lambda row: calculate_key_delays(row["press_times_2"], row["release_times_2"]),
        axis=1
    )

    df['all_key_delays'] = df.apply(
        lambda row: np.concatenate([
            row['key_delays_1'] if isinstance(row['key_delays_1'], np.ndarray) else np.array([]),
            row['key_delays_2'] if isinstance(row['key_delays_2'], np.ndarray) else np.array([])
        ]),
        axis=1
    )

    exploded = df[['has_PD', 'all_key_delays']].explode('all_key_delays')
    exploded = exploded.dropna(subset=['all_key_delays'])
    exploded['all_key_delays'] = exploded['all_key_delays'].astype(float)

    pt = exploded.groupby('has_PD')['all_key_delays'].agg(['sum', 'count', 'mean']).rename(
        columns={'sum': 'total_key_delays', 'count': 'n_key_delays', 'mean': 'avg_key_delay'}
    )
    return pt

pt = calculate_average_key_delays(all_trials).reset_index()
ax = px.bar(pt, x='has_PD', y='avg_key_delay')
ax.update_layout(
    xaxis_title='Has Parkinson\'s Disease',
    yaxis_title='Average Key Delay',
    title='Average Key Delay vs. Has PD'
)
ax.show()


# %% [markdown]
# Graph 2 - Key Delay vs. UPDRS-III
# %%

def calculate_key_delays_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    df['key_delays_1'] = df.apply(
        lambda row: calculate_key_delays(row["press_times_1"], row["release_times_1"]),
        axis=1
    )
    df['key_delays_2'] = df.apply(
        lambda row: calculate_key_delays(row["press_times_2"], row["release_times_2"]),
        axis=1
    )
    df['all_key_delays'] = df.apply(
        lambda row: np.concatenate([
            row['key_delays_1'] if isinstance(row['key_delays_1'], np.ndarray) else np.array([]),
            row['key_delays_2'] if isinstance(row['key_delays_2'], np.ndarray) else np.array([])
        ]),
        axis=1
    )

    exploded = df[['pID', 'UPDRS-III', 'all_key_delays']].explode('all_key_delays')
    exploded = exploded.dropna(subset=['all_key_delays'])
    exploded['all_key_delays'] = exploded['all_key_delays'].astype(float)
    pt = exploded.groupby(['pID', 'UPDRS-III'])['all_key_delays'].mean()
    return pt

pt = calculate_key_delays_per_patient(all_trials).reset_index()
ax = px.scatter(pt, x='UPDRS-III', y='all_key_delays')
ax.update_layout(
    xaxis_title='UPDRS-III',
    yaxis_title='Average Key Delay',
    title='Average Key Delay vs. UPDRS-III'
)
ax.show()

# %% [markdown]
# # Graph 3: Correlation between UPDRS-III and Alternating Finger Tapping Scores
# %%

def plot_updrs_alternating_tapping_correlation(df: pd.DataFrame) -> None:
    # Plotly scatter plot
    fig = px.scatter(
        df,
        x="UPDRS-III",
        y="alternating_finger_tapping",
        color="has_PD",
        title="Correlation between UPDRS-III and Alternating Finger Tapping Scores",
        labels={
            "UPDRS-III": "UPDRS-III Score",
            "alternating_finger_tapping": "Alternating Finger Tapping Score",
            "has_PD": "Parkinson's Disease Status"
        },
        trendline="ols",  # Add linear regression line
        color_discrete_sequence=["#2ecc71", "#e74c3c"]  # Green for non-PD, Red for PD
    )

    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16
    )

    fig.write_image(ASSETS_PATH / "updrs_alternating_tapping_correlation.png")

plot_updrs_alternating_tapping_correlation(all_trials)

# %% [markdown]
# # Graph 4: Typing Speed Comparison Between Groups
# %%

def plot_typing_speed_comparison(df: pd.DataFrame) -> None:
    # Plotly combined violin and box plot
    fig = px.violin(
        df,
        x="has_PD",
        y="typing_speed",
        color="has_PD",
        box=True,  # Add box plot inside violin
        points="all",  # Show all points
        title="Typing Speed Distribution: Parkinson's vs Control",
        labels={
            "has_PD": "Group",
            "typing_speed": "Typing Speed",
            "has_PD": "Parkinson's Disease Status"
        },
        color_discrete_sequence=["#2ecc71", "#e74c3c"]  # Green for non-PD, Red for PD
    )

    # Update x-axis labels
    fig.update_xaxes(
        ticktext=["Control", "Parkinson's"],
        tickvals=[0, 1]
    )

    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        showlegend=False  # Hide legend since colors are self-explanatory
    )

    fig.write_image(ASSETS_PATH / "typing_speed_comparison.png")

plot_typing_speed_comparison(all_trials)

# %% [markdown]
# # Graph 5 - Distribution of nQi Scores by PD Status
# %%

plt.figure(figsize=(8, 5))
for label, group in all_trials.groupby("has_PD"):
    plt.hist(group["nQi"].dropna(), bins=20, alpha=0.5, label=f"PD = {label}")

plt.xlabel("nQi Score")
plt.ylabel("Frequency")
plt.title("Distribution of nQi Scores by PD Status")
plt.legend()
plt.tight_layout()
plt.savefig(ASSETS_PATH / "nqi_histogram_by_pd.png", dpi=300)
plt.show()

# %% [markdown]
# # Graph 6 - Average Variance of Hold Times by PD Status
# %%

all_trials["hold_var_1"] = all_trials["hold_times_1"].apply(
    lambda x: np.var(x) if isinstance(x, np.ndarray) and len(x) > 1 else np.nan
)
all_trials["hold_var_2"] = all_trials["hold_times_2"].apply(
    lambda x: np.var(x) if isinstance(x, np.ndarray) and len(x) > 1 else np.nan
)

all_trials["avg_hold_var"] = all_trials[["hold_var_1", "hold_var_2"]].mean(axis=1)

fig, ax = plt.subplots()
all_trials.boxplot(column="avg_hold_var", by="has_PD", ax=ax)
ax.set_title("Average Variance of Hold Times by PD Status")
ax.set_xlabel("Has Parkinson's Disease")
ax.set_ylabel("Hold Time Variance")
plt.suptitle("")
fig.tight_layout()
fig.savefig(ASSETS_PATH / "hold_time_variance_vs_pd.png", dpi=300)
plt.show()