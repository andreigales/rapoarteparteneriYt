import streamlit as st
import pandas as pd

# =========================
# Settings (to avoid crashes)
# =========================
MAX_CSV_FILES = 12
CHUNK_SIZE = 200_000  # lower if you still hit RAM issues (e.g., 50_000)


# =================
# Helper Functions
# =================
def load_asset_ids(excel_file):
    """Load asset IDs from the uploaded Excel file (first column)."""
    try:
        df = pd.read_excel(excel_file, header=None, dtype=str)
        asset_ids = (
            df.iloc[:, 0]
            .astype(str)
            .str.replace("\xa0", " ")
            .str.strip()
        )
        return {aid for aid in asset_ids if aid and aid.lower() != "nan"}
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return set()


def process_standard_csv(file, asset_ids):
    """Standard report: chunk-read CSV, filter by asset_id column index 3, keep desired cols."""
    DESIRED_COLS = [1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 26, 27]

    # Make sure stream is at start
    try:
        file.seek(0)
    except Exception:
        pass

    results = []
    try:
        reader = pd.read_csv(
            file,
            header=None,
            dtype=str,
            low_memory=False,
            chunksize=CHUNK_SIZE,
        )
    except Exception as e:
        st.warning(f"Skipping {file.name}: {e}")
        return None

    for chunk in reader:
        # Need at least col 3
        if chunk.shape[1] <= 3:
            continue

        # Clean asset ID column
        asset_col_idx = 3
        chunk[asset_col_idx] = (
            chunk[asset_col_idx]
            .astype(str)
            .str.replace("\xa0", " ")
            .str.strip()
        )

        filtered = chunk[chunk[asset_col_idx].isin(asset_ids)]
        if filtered.empty:
            continue

        cols = filtered.shape[1]
        valid_cols = [i for i in DESIRED_COLS if i < cols]
        sub = filtered[valid_cols].copy()
        sub["source_file"] = file.name
        results.append(sub)

    if not results:
        return None

    return pd.concat(results, ignore_index=True)


def process_redlabel_csv(file, asset_ids):
    """Red Label report: read header names, then chunk-read data and filter."""
    DESIRED_COLS = [1, 2, 3, 4, 6, 9, 11, 12, 13, 17, 26]

    try:
        file.seek(0)
    except Exception:
        pass

    try:
        # Header line is at row 2 (skiprows=1), read only header columns
        hdr_df = pd.read_csv(file, nrows=0, skiprows=1)
        header_names = list(hdr_df.columns)

        # Reset pointer for data reading
        file.seek(0)

        reader = pd.read_csv(
            file,
            header=None,
            skiprows=2,
            dtype=str,
            chunksize=CHUNK_SIZE,
        )

    except Exception as e:
        st.warning(f"Error preparing {file.name}: {e}")
        return None

    results = []
    for chunk in reader:
        if chunk.shape[1] <= 3:
            continue

        asset_col_idx = 3
        chunk[asset_col_idx] = (
            chunk[asset_col_idx]
            .astype(str)
            .str.replace("\xa0", " ")
            .str.strip()
        )

        filtered = chunk[chunk[asset_col_idx].isin(asset_ids)]
        if filtered.empty:
            continue

        cols = filtered.shape[1]
        valid_indices = [i for i in DESIRED_COLS if i < cols]

        # Map indices to header names (only if header exists for those indices)
        good_headers = []
        for i in valid_indices:
            if i < len(header_names):
                good_headers.append(header_names[i])
            else:
                good_headers.append(f"col_{i}")

        sub = filtered[valid_indices].copy()
        sub.columns = good_headers
        sub["source_file"] = file.name
        results.append(sub)

    if not results:
        return None

    return pd.concat(results, ignore_index=True)


def insert_platform_currency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    - After column E (5th col), insert a column filled with 'YouTube'
    - After column J (10th col), insert a column filled with 'USD'

    Note: This is based on the *current output columns* (A..), not original CSV letters.
    """
    out = df.copy()

    # After E => insert at position 5 (0-based insert index 5)
    pos_youtube = min(5, len(out.columns))
    out.insert(pos_youtube, "Platform", "YouTube")

    # After J => insert at position 10 (0-based insert index 10)
    # (this is after the 10th column in the current sheet)
    pos_usd = min(10, len(out.columns))
    out.insert(pos_usd, "Currency", "USD")

    return out


# ==========
# Main UI
# ==========
st.set_page_config(page_title="YouTube Asset Reporter", layout="wide")

st.title("📊 YouTube Asset Report Generator")
st.markdown("Upload your asset list and raw CSV files to generate the report.")

report_type = st.sidebar.radio("Select Report Type", ["Standard Report", "Red Label Report"])

st.subheader("1. Upload Inputs")
uploaded_excel = st.file_uploader("Upload 'assets soundfeed.xlsx'", type=["xlsx", "xls"])
uploaded_csvs = st.file_uploader("Upload Raw CSV Files", type=["csv"], accept_multiple_files=True)

if uploaded_csvs and len(uploaded_csvs) > MAX_CSV_FILES:
    st.error(f"Poți încărca maxim {MAX_CSV_FILES} fișiere CSV. Ai selectat {len(uploaded_csvs)}.")
    st.stop()

if st.button("Generate Report"):
    if not uploaded_excel:
        st.error("Please upload the Asset Excel file first.")
    elif not uploaded_csvs:
        st.error("Please upload at least one CSV file.")
    else:
        with st.spinner("Processing files..."):
            asset_ids = load_asset_ids(uploaded_excel)
            st.success(f"Loaded {len(asset_ids)} asset IDs.")

            if not asset_ids:
                st.error("Lista de asset IDs este goală (sau nu a putut fi citită).")
                st.stop()

            all_results = []
            progress_bar = st.progress(0)

            for i, file in enumerate(uploaded_csvs):
                if report_type == "Standard Report":
                    res = process_standard_csv(file, asset_ids)
                else:
                    res = process_redlabel_csv(file, asset_ids)

                if res is not None and not res.empty:
                    all_results.append(res)

                progress_bar.progress((i + 1) / len(uploaded_csvs))

            if all_results:
                final_df = pd.concat(all_results, ignore_index=True, sort=False)

                # Insert requested columns
                final_df = insert_platform_currency_columns(final_df)

                st.write(f"### Found {len(final_df)} matching rows")
                st.dataframe(final_df.head(50))

                csv_bytes = final_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 Download Final Report CSV",
                    data=csv_bytes,
                    file_name=f"report_{report_type.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No matching asset IDs found in any of the uploaded files.")
