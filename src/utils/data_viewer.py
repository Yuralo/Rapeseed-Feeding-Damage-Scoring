"""Browser-based viewer for the RSFB leaf-damage data set.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
from utils.image import detect_grid, image_to_ndarray, to_rgb, warp_big_square
import streamlit as st
from PIL import Image
from utils.data import image_path

DEFAULT_DATASET_PATH = (
    "/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage/"
    "RSFB-Phenotyping_training_set/RSFB-Phenotyping_training_set"
)
SCORES_FILE = "RSFB-Phenotyping_training_set_scores.csv"


@st.cache_data
def load_data(dataset_path: str) -> pd.DataFrame:
    """Load and validate the image-score table."""
    data = pd.read_csv(Path(dataset_path) / SCORES_FILE)
    required_columns = {"Filename", "mean_score"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"The CSV is missing column(s): {', '.join(sorted(missing))}")
    return data


st.set_page_config(page_title="Leaf Damage Viewer", layout="wide")
st.title("Leaf Damage Image Viewer")

dataset_path_string = st.sidebar.text_input("Dataset folder", DEFAULT_DATASET_PATH)
dataset_path = Path(dataset_path_string)

try:
    data = load_data(dataset_path_string)
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load the scores file: {error}")
    st.stop()

if data.empty:
    st.warning("The scores file contains no images.")
    st.stop()

if "image_index" not in st.session_state:
    st.session_state.image_index = 0

last_index = len(data) - 1
st.session_state.image_index = min(st.session_state.image_index, last_index)

previous, position, following = st.columns([1, 2, 1])
with previous:
    if st.button("← Back", disabled=st.session_state.image_index == 0, use_container_width=True):
        st.session_state.image_index -= 1
        st.rerun()
with following:
    if st.button("Forward →", disabled=st.session_state.image_index == last_index, use_container_width=True):
        st.session_state.image_index += 1
        st.rerun()
with position:
    st.markdown(f"<div style='text-align: center'>Image {st.session_state.image_index + 1} of {len(data)}</div>", unsafe_allow_html=True)

row = data.iloc[st.session_state.image_index]
path = image_path(dataset_path, str(row["Filename"]))

left, right = st.columns([3, 1])
with left:
    if path.is_file():
        image = image_to_ndarray(path)
        grid_points, _, _ = detect_grid(image)
        big_square = warp_big_square(
            image,
            grid_points,
        )
        image = to_rgb(big_square)
        st.image(image, caption=path.name, use_container_width=True)
    else:
        st.error(f"Image not found: {path}")
with right:
    st.subheader("Label")
    st.metric("Mean score", row["mean_score"])
    st.caption(f"Filename: {row['Filename']}")
