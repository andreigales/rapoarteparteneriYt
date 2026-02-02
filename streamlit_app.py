import streamlit as st
import pandas as pd

# --- Helper Functions ---
def load_asset_ids(excel_file):
    """Load asset IDs from the uploaded Excel file."""
    try:
        df = pd.read_excel(excel_file, header=None, dtype=str)
        asset_ids = (
            df.iloc[:, 0].astype(str).str.replace('\xa0', ' ').str.strip()
        )
        return {aid for aid in asset_ids if aid}
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return set()

def process_standard_csv(file, asset_ids):
    """Logic from report_generator_assets.py"""
    DESIRED_COLS = [1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 26, 27]
    try:
        df = pd.read_csv(file, header=None, dtype=str, low_memory=False)
    except Exception as e:
        st.warning(f"Skipping {file.name}: {e}")
        return None
    rows, cols = df.shape
    if cols <= 3:
        return None
    asset_col_idx = 3
    df[asset_col_idx] = df[asset_col_idx].astype(str).str.replace('\xa0', ' ').str.strip()
    filtered = df[df[asset_col_idx].isin(asset_ids)]
    if filtered.empty:
        return None
    valid_cols = [i for i in DESIRED_COLS if i < cols]
    sub = filtered[valid_cols].copy()
    sub['source_file'] = file.name
    return sub

def process_redlabel_csv(file, asset_ids):
    """Logic from report_generator_assets_redlabel.py"""
    DESIRED_COLS = [1, 2, 3, 4, 6, 9, 11, 12, 13, 17, 26]
    file.seek(0)
    try:
        hdr_df = pd.read_csv(file, nrows=0, skiprows=1)
        header_names = list(hdr_df.columns)
        file.seek(0)
        df = pd.read_csv(file, header=None, skiprows=2, dtype=str)
        rows, cols = df.shape
        if cols <= 3:
            return None
        asset_col_idx = 3
        df[asset_col_idx] = df[asset_col_idx].astype(str).str.replace("\xa0", " ").str.strip()
        filtered = df[df[asset_col_idx].isin(asset_ids)]
        if filtered.empty:
            return None
        valid_indices = [i for i in DESIRED_COLS if i < cols]
        good_headers = [header_names[i] for i in valid_indices]
        sub = filtered[valid_indices].copy()
        sub.columns = good_headers
        sub["source_file"] = file.name
        return sub
    except Exception as e:
        st.warning(f"Error processing {file.name}: {e}")
        return None

# --- Main App UI ---
st.set_page_config(page_title="YouTube Asset Reporter", layout="wide")
st.title(" YouTube Asset Report Generator")
st.markdown("Upload your asset list and raw CSV files to generate the report.")

report_type = st.sidebar.radio("Select Report Type", ["Standard Report", "Red Label Report"])

st.subheader("1. Upload Inputs")
uploaded_excel = st.file_uploader("Upload 'assets soundfeed.xlsx'", type=["xlsx", "xls"])
uploaded_csvs = st.file_uploader("Upload Raw CSV Files", type=["csv"], accept_multiple_files=True)

if st.button("Generate Report"):
    if not uploaded_excel:
        st.error("Please upload the Asset Excel file first.")
    elif not uploaded_csvs:
        st.error("Please upload at least one CSV file.")
    else:
        with st.spinner("Processing files..."):
            asset_ids = load_asset_ids(uploaded_excel)
            st.success(f"Loaded {len(asset_ids)} asset IDs.")
            all_results = []
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_csvs):
                if report_type == "Standard Report":
                    res = process_standard_csv(file, asset_ids)
                else:
                    res = process_redlabel_csv(file, asset_ids)
                if res is not None:
                    all_results.append(res)
                progress_bar.progress((i + 1) / len(uploaded_csvs))
            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                st.write(f"### Found {len(final_df)} matching rows")
                st.dataframe(final_df.head())
                csv = final_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label=" Download Final Report CSV",
                    data=csv,
                    file_name=f"report_{report_type.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No matching asset IDs found in any of the uploaded files.")
