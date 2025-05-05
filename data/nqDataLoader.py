# -*- coding: utf-8 -*-

"""
NqDataLoader - Module for loading and filtering keyboard data.
"""

import numpy as np
import os
import re
import datetime
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for maximum hold time
MAX_HOLD_TIME = 5


class NqDataLoader:
    """Class for loading and processing keyboard data files."""

    # Filter constants
    FLT_NO_MOUSE = 1 << 0
    FLT_NO_LETTERS = 1 << 1
    FLT_NO_BACK = 1 << 2
    FLT_NO_SHORT_META = 1 << 3  # space, enter, arrows, etc.
    FLT_NO_LONG_META = 1 << 4  # shift, control, alt, etc.
    FLT_NO_PUNCT = 1 << 5

    def __init__(self):
        """Initialize data loader with empty data structures."""
        self.data_keys = None
        self.data_ht = None
        self.data_time_start = None
        self.data_time_end = None
        self.data_ft = None
        self.lbl = None

    def sanity_check(self) -> int:
        """
        Filter out keystrokes variables in the member variables.
        Eliminate anything < 0.

        Returns:
            int: Number of elements removed
        """
        assert self.data_keys is not None and len(self.data_keys) > 0
        assert self.data_ht is not None and len(self.data_ht) > 0
        assert self.data_time_start is not None and len(self.data_time_start) > 0
        assert self.data_time_end is not None and len(self.data_time_end) > 0

        # Create mask for invalid values
        bad_lbl = self.data_time_start <= 0
        bad_lbl = np.bitwise_or(bad_lbl, self.data_time_end <= 0)
        bad_lbl = np.bitwise_or(bad_lbl, self.data_ht < 0)
        bad_lbl = np.bitwise_or(bad_lbl, self.data_ht >= MAX_HOLD_TIME)

        # Remove non-consecutive start times
        non_cons_tmp_lbl = np.ones(
            len(self.data_time_start), dtype=bool
        )  # Start with all True labels
        non_cons_lbl = np.zeros(
            len(self.data_time_start), dtype=bool
        )  # Start with all False labels
        start_tmp_arr = self.data_time_start.copy()

        while np.sum(non_cons_tmp_lbl) > 0:
            # Find non-consecutive labels
            non_cons_tmp_lbl = np.append([False], np.diff(start_tmp_arr) < 0)
            # Keep track of the indices to remove
            non_cons_lbl = np.bitwise_or(non_cons_lbl, non_cons_tmp_lbl)
            # Changes value in the temporary array
            indices_to_change = np.arange(len(non_cons_tmp_lbl))[non_cons_tmp_lbl]
            start_tmp_arr[indices_to_change] = start_tmp_arr[indices_to_change - 1]

        bad_lbl = np.bitwise_or(bad_lbl, non_cons_lbl)

        # Invert bad labels
        good_lbl = np.bitwise_not(bad_lbl)

        # Filter data
        self.data_keys = self.data_keys[good_lbl]
        self.data_ht = self.data_ht[good_lbl]
        self.data_time_start = self.data_time_start[good_lbl]
        self.data_time_end = self.data_time_end[good_lbl]

        return int(np.sum(bad_lbl))

    def load_data_file(
        self,
        file_in: str,
        auto_filt: bool = True,
        imp_type: str = None,
        debug: bool = False,
    ):
        """
        Load raw data file.

        Args:
            file_in: Path to the input file
            auto_filt: Whether to automatically filter data
            imp_type: Import type ('si' for sleep inertia format)
            debug: Whether to print debug information

        Returns:
            bool or str: True if successful, error message if failed
        """
        try:
            if imp_type == "si":  # Sleep inertia format
                data = np.genfromtxt(file_in, dtype=int, delimiter=",", skip_header=0)
                data = data - data.min()
                data = data.astype(np.float64) / 1000
                self.data_time_start = data[:, 0]
                self.data_time_end = data[:, 1]
                self.data_ht = self.data_time_end - self.data_time_start
                # Just to make sanity check work
                self.data_keys = np.zeros(len(self.data_ht))
                rem_num = self.sanity_check()
                if debug:
                    logger.info(f"Removed {rem_num} elements")
            else:  # PD format
                data = np.genfromtxt(
                    file_in, dtype=None, delimiter=",", skip_header=0, encoding="utf-8"
                )
                # Load data fields
                self.data_keys = data["f0"]
                self.data_ht = data["f1"]
                self.data_time_start = data["f3"]  # Changed order from 2<->3
                self.data_time_end = data["f2"]
                rem_num = self.sanity_check()

                if debug:
                    logger.info(
                        f"Removed {rem_num} elements ({100.0 * rem_num / len(self.data_ht):.2f}%)"
                    )

                if auto_filt:
                    self.filt_data(self.FLT_NO_MOUSE | self.FLT_NO_LONG_META)

            # Calculate flight time
            self.data_ft = np.array(
                [
                    self.data_time_start[i] - self.data_time_start[i - 1]
                    for i in range(1, self.data_time_start.size)
                ]
            )
            self.data_ft = np.append(self.data_ft, 0)

            return True
        except IOError:
            error_str = f"File {file_in} not found"
            logger.error(error_str)
            return error_str

    def load_data_arr(self, lst_arr: list[str]) -> None:
        """
        Load data from a list of strings.

        Args:
            lst_arr: List of comma-separated data strings
        """
        self.data_keys = np.zeros((len(lst_arr), 1), dtype="S30")
        self.data_ht = np.zeros((len(lst_arr), 1))
        self.data_time_start = np.zeros((len(lst_arr), 1))
        self.data_time_end = np.zeros((len(lst_arr), 1))

        for i, row in enumerate(lst_arr):
            tok = row.split(",")
            self.data_keys[i] = str(tok[0])
            self.data_ht[i] = float(tok[1])
            self.data_time_start[i] = float(tok[2])
            self.data_time_end[i] = float(tok[3])

    def filt_data(self, flags: int) -> None:
        """
        Filter data according to specified flags.

        Args:
            flags: Bit flags for filtering (use FLT_* constants)
        """
        # Define filters
        p_mouse = re.compile(r'("mouse.+")')
        p_char = re.compile(r'(".{1}")')
        p_back = re.compile(r'("BackSpace")')
        p_long_meta = re.compile(r'("Shift.+")|("Alt.+")|("Control.+")')
        p_short_meta = re.compile(
            r'("space")|("Num_Lock")|("Return")|("P_Enter")|'
            r'("Caps_Lock")|("Left")|("Right")|("Up")|("Down")'
        )
        p_punct = re.compile(
            r'("more")|("less")|("exclamdown")|("comma")|'
            r'("\[65027\]")|("\[65105\]")|("ntilde")|("minus")|("equal")|'
            r'("bracketleft")|("bracketright")|("semicolon")|("backslash")|'
            r'("apostrophe")|("comma")|("period")|("slash")|("grave")'
        )

        # Create mask labels
        lbl = np.ones(len(self.data_keys), dtype=bool)

        if flags & self.FLT_NO_MOUSE:
            lbl_tmp = np.array([p_mouse.match(k) is None for k in self.data_keys])
            lbl = np.logical_and(lbl, lbl_tmp)

        if flags & self.FLT_NO_LETTERS:
            lbl_tmp = np.array([p_char.match(k) is None for k in self.data_keys])
            lbl = np.logical_and(lbl, lbl_tmp)

        if flags & self.FLT_NO_BACK:
            lbl_tmp = np.array([p_back.match(k) is None for k in self.data_keys])
            lbl = np.logical_and(lbl, lbl_tmp)

        if flags & self.FLT_NO_SHORT_META:
            lbl_tmp = np.array([p_short_meta.match(k) is None for k in self.data_keys])
            lbl = np.logical_and(lbl, lbl_tmp)

        if flags & self.FLT_NO_LONG_META:
            lbl_tmp = np.array([p_long_meta.match(k) is None for k in self.data_keys])
            lbl = np.logical_and(lbl, lbl_tmp)

        if flags & self.FLT_NO_PUNCT:
            lbl_tmp = np.array([p_punct.match(k) is None for k in self.data_keys])
            lbl = np.logical_and(lbl, lbl_tmp)

        # Store and apply the filter
        self.lbl = lbl
        self.data_keys = self.data_keys[lbl]
        self.data_ht = self.data_ht[lbl]
        self.data_time_start = self.data_time_start[lbl]
        self.data_time_end = self.data_time_end[lbl]

    @classmethod
    def get_std_variables_filt(
        cls, file_in: str, imp_type: str = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get filtered variables from a file.

        Args:
            file_in: Path to the raw typing file
            imp_type: Format of the input file ('si' for sleep inertia data)

        Returns:
            Tuple of arrays: (keys, hold times, press timestamps, release timestamps)
        """
        nq_obj = cls()
        res = nq_obj.load_data_file(file_in, False, imp_type)
        # Remove delete button
        nq_obj.filt_data(
            nq_obj.FLT_NO_MOUSE | nq_obj.FLT_NO_LONG_META | nq_obj.FLT_NO_BACK
        )
        assert res is True, "File not found or error loading data"

        return (
            nq_obj.data_keys,
            nq_obj.data_ht,
            nq_obj.data_time_start,
            nq_obj.data_time_end,
        )


def get_data_filt_helper(
    file_in: str, imp_type: str = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper method to load filtered keypress data from given file.

    Args:
        file_in: Path to csv keypress file
        imp_type: Format of the csv file ('si' for sleep inertia data, None for PD data)

    Returns:
        Tuple of arrays: (keys, hold times, press timestamps, release timestamps)
    """
    nq_obj = NqDataLoader()
    res = nq_obj.load_data_file(file_in, False, imp_type)
    # Remove delete button
    nq_obj.filt_data(nq_obj.FLT_NO_MOUSE | nq_obj.FLT_NO_LONG_META | nq_obj.FLT_NO_BACK)
    assert res is True, "File not found or error loading data"

    return (
        nq_obj.data_keys,
        nq_obj.data_ht,
        nq_obj.data_time_start,
        nq_obj.data_time_end,
    )


def gen_file_struct(data_dir: str, max_rep_num: int = 4) -> tuple[dict, dict]:
    """
    Generate a dictionary with the NQ file list and test date (legacy method).

    Args:
        data_dir: Base directory containing the CSV files
        max_rep_num: Integer with the maximum repetition number

    Returns:
        Two dictionaries: fMap, dateMap = NQ file/date list[pID][repID][expID]
    """
    f_map = {}  # data container
    date_map = {}
    files = os.listdir(data_dir)
    p = re.compile(r"([0-9]+)\.{1}([0-9]+)_([0-9]+)_([0-9]+)\.csv")

    for f in files:
        m = p.match(f)

        if m:  # file found
            time_stamp = m.group(1)
            p_id = int(m.group(2))
            rep_id = int(m.group(3))
            exp_id = int(m.group(4))

            # Store new patient
            if p_id not in f_map:
                f_map[p_id] = {}
                date_map[p_id] = {}
                for tmp_rid in range(1, max_rep_num + 1):
                    f_map[p_id][tmp_rid] = {}
                    date_map[p_id][tmp_rid] = {}

            # Store data
            f_map[p_id][rep_id][exp_id] = data_dir + f
            date_map[p_id][rep_id][exp_id] = datetime.datetime.fromtimestamp(
                int(time_stamp)
            )
        else:
            logger.debug(f"File {f} doesn't match the expected pattern")

    return f_map, date_map


if __name__ == "__main__":
    # Example usage
    logger.info("NqDataLoader module - use as library or import specific functions")
