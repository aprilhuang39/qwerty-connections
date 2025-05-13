"""
This script is used to generate the data for the force graph.

Nodes:
- Patients (pID)
Color:
- Has PD: red
- No PD: blue
Size:
- UPDRS-III

Edges:
- Euclidian distance betweeen patients based on (nQI, typing speed, alternating tapping, single tapping, etc.)

Pipeline:
1. Load data
2. Calculate all distances for option permutations
3. Generate force graph data as such:
    - nodes: pID, has_PD, UPDRS-III (stored as dictionary and exported to json)
    - edges: euclidean distance between patients based on option permutations (stored as Apache Arrow Table in a dataframe and exported as parquet for efficient column-wise access)
"""

import json
import pandas as pd
import numpy as np
import data.nqDataLoader as nq
from pathlib import Path

# Data Loading
DATA_PATH = Path("data")
ASSETS_PATH = Path("docs/force_graph")

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

def generate_option_permutations(all_trials: pd.DataFrame) -> list[str]:
    """
    Generate the option permutations for the force graph. Note: these are hand-picked.
    """
    return [['nQi', 'typing_speed'],
            # ['nQi', 'typing_speed', 'alternating_finger_tapping', 'single_finger_tapping'],
            # ['nQi', 'typing_speed', 'alternating_finger_tapping'],
            # ['nQi', 'typing_speed', 'single_finger_tapping'],
            # ['nQi', 'alternating_finger_tapping', 'single_finger_tapping'],
            # ['nQi', 'alternating_finger_tapping'],
            # ['nQi', 'single_finger_tapping'],
    ]

def generate_node_data(all_trials: pd.DataFrame) -> list[dict]:
    """
    Generate the node data for the force graph.
    - pID: int
    - has_PD: boolean
    - UPDRS-III: float
    """
    return all_trials[['pID', 'has_PD', 'UPDRS-III']].to_dict(orient='records')

def generate_edge_data(all_trials: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Calculate the euclidean distance between patients based on the given metrics.

    Args:
        all_trials: DataFrame containing patient data with relevant metrics
        metrics: List of column names to use for distance calculation

    Returns:
        DataFrame with distances between all pairs of patients
    """
    # Extract patient IDs and metric values
    patients = all_trials['pID'].values
    n_patients = len(patients)
    n_metrics = len(metrics)

    # Normalize the metric values to ensure fair comparison
    metric_values = all_trials[metrics].copy()
    for metric in metrics:
        if metric_values[metric].std() > 0:  # Avoid division by zero
            metric_values[metric] = (metric_values[metric] - metric_values[metric].mean()) / metric_values[metric].std()
        else:
            metric_values[metric] = 0

    # Create a DataFrame to store distances
    distances = []

    # Calculate Euclidean distance between each pair of patients
    for i in range(n_patients):
        for j in range(i+1, n_patients):  # Only calculate each pair once
            patient_i = patients[i]
            patient_j = patients[j]

            # Extract metric values for both patients
            values_i = metric_values.iloc[i][metrics].values
            values_j = metric_values.iloc[j][metrics].values

            # Calculate Euclidean distance
            squared_diff_sum = np.sum((values_i - values_j) ** 2)

            # Normalize by taking the nth root where n is the number of metrics
            if n_metrics > 0:
                distance = squared_diff_sum ** (1.0 / n_metrics)
            else:
                distance = 0.0

            # Store the result
            distances.append({
                'pID_1': patient_i,
                'pID_2': patient_j,
                'distance': distance
            })

    return pd.DataFrame(distances)

def export_node_data(nodes: dict, export_dir: Path) -> None:
    """
    Export the force graph data to a json file (nodes).
    """
    with open(export_dir / "nodes.json", "w") as f:
        json.dump(nodes, f)

def export_edge_data(edges: pd.DataFrame, export_dir: Path, metric_permutation: list[str]) -> None:
    """
    Export the force graph data to a parquet file (edges).
    """
    edges.to_parquet(export_dir / f"edges_{'_'.join(metric_permutation)}.parquet")

if __name__ == "__main__":
    ground_truth_1 = load_ground_truth("MIT-CS1PD")
    ground_truth_2 = load_ground_truth("MIT-CS2PD")
    ground_truth = pd.concat([ground_truth_1, ground_truth_2])

    all_trials = load_all_trials(ground_truth)

    metric_permutations = generate_option_permutations(all_trials)

    export_dir = ASSETS_PATH / "force_graph"
    export_dir.mkdir(parents=True, exist_ok=True)

    nodes = generate_node_data(all_trials)
    export_node_data(nodes, export_dir)

    for metric_permutation in metric_permutations:
        distances = generate_edge_data(all_trials, metric_permutation)
        export_edge_data(distances, export_dir, metric_permutation)
