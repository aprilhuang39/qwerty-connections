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
# # Graph 1
# %%


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