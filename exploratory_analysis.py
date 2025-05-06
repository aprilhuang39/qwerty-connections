# %%
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import data.nqDataLoader as nq
from pathlib import Path

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
# # Graph 1 - Average Key Delay vs. UPDRS-III
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

pt = calculate_average_key_delays(all_trials)
ax = pt['avg_key_delay'].plot.bar()
ax.set_xlabel('Has Parkinson\'s Disease')
ax.set_ylabel('Average Key Delay')
ax.set_title('Average Key Delay vs. Has PD')
ax.figure.savefig(ASSETS_PATH / "avg_key_delay_vs_has_pd.png", dpi=300)
plt.show()

# %% [markdown]
# Graph 2
# %%

pass

# %% [markdown]
# # Graph 3
# %%

pass

# %% [markdown]
# # Graph 4
# %%

pass

# %% [markdown]
# # Graph 5
# %%

pass

# %% [markdown]
# # Graph 6
# %%

pass