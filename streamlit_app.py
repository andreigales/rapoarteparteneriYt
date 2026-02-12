import streamlit as st
import pandas as pd
from pandas.errors import ParserError

# =========================
# Settings (stability)
# =========================
MAX_CSV_FILES = 12
CHUNK_SIZE = 200_000  # reduce (e.g. 50_000) if still crashing on RAM


# =========================
# Helper: robust chunk reader
# =========================
def read_csv_chunks_robust(
    uploaded_file,
    header=None,
    skiprows=0,
    dtype=str,
    low_memory=False,
    chunksize=CHUNK_SIZE,
):
    """
    Returns an iterator of chunks for a CSV file.

    Strategy:
    1) Fast C engine + on_bad_lines='skip' (works for many malformed lines)
    2) If parse fails (including mid-iteration), fallback:
       python engine + sep=None (auto-detect delimiter) + on_bad_lines='skip'
    """
    def _reader(engine, sep):
        uploaded_file.seek(0)
        return pd.read_csv(
            uploaded_file,
            header=header,
            skiprows=skiprows,
            dtype=dtype,
            low_memory=low_memory,
            chunksize=chunksize,
            engine=engine,
            sep=sep,
            on_bad_lines="skip",  # critical for broken rows
        )

    # Try fast path first
    try:
        return _reader(engine="c", sep=",")
    except Exception:
        # immediate fallback if even constructing the reader fails
        return _reader(engine="python", sep=None)


def iter_chunks_with_fallback(uploaded_file, header=None, skiprows=0):
    """
    Iterates chunks, and if a ParserError happens during iteration,
    retries the entire file with the python auto-delimiter engine.
    """
    # First attempt
    reader = read_csv_chunks_robust(uploaded_file, header=header, skiprows=skiprows)

    try:
        for chunk in reader:
            yield chunk
        return
    except ParserError:
        # Fallback (full restart, more tolerant)
        uploaded_file.seek(0)
        reader2 = pd.read_csv(
            uploaded_file,
            header=header,
            skiprows=skiprows,
            dtype=str,
            low_memory=False,
            chunksize=CHUNK_SIZE,
            engine="python",
            sep=None,              # auto-detect delimiter
            on_bad_lines="skip",   # skip broken lines
        )
        for chunk in reader2:
            yield chunk


# =================
# Business Helpers
# =================
def load_asset_ids(excel_file):
    """Load asset IDs from first column of uploaded Excel file."""
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
    """Standard report: chunk-read, filter by asset_id column index 3, keep desired cols."""
    DESIRED_COLS = [1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 26, 27]
    asset_col_idx = 3

    results = []

    try:
        for chunk in iter_chunks_with_fallback(file, header=None, skiprows=0):
            if chunk.shape[1] <= asset_col_idx:
                continue

            # Clean asset id column
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

    except Exception as e:
        st.warning(f"Error reading {file.name}: {e}")
        return None

    if not results:
        return None

    return pd.concat(results, ignore_index=True)


def process_redlabel_csv(file, asset_ids):
    """Red Label report: read header names (row 2), then chunk-read data from row 3 onward."""
    DESIRED_COLS = [1, 2, 3, 4, 6, 9, 11, 12, 13, 17, 26]
    asset_col_idx = 3

    # Read header names (row 2 => skiprows=1, take columns from header)
    try:
        file.seek(0)
        hdr_df = pd.read_csv(
            file,
            nrows=0,
            skiprows=1,
            engine="python",
            sep=None,
            on_bad_lines="skip",
        )
        header_names = list(hdr_df.columns)
    except Exception as e:
        st.warning(f"Could not read header for {file.name} (Red Label): {e}")
        header_names = []

    results = []

    try:
        # Data starts after 2 rows
        for chunk in iter_chunks_with_fallback(file, header=None, skiprows=2):
            if chunk.shape[1] <= asset_col_idx:
                continue

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

            sub = filtered[valid_indices].copy()

            # Apply headers where possible
            new_cols = []
            for i in valid_indices:
                if i < len(header_names):
                    new_cols.append(header_names[i])
                else:
                    new_cols.append(f"col_{i}")
            sub.columns = new_cols

            sub["source_file"] = file.name
            results.append(sub)

    except Exception as e:
        st.warning(f"Error processing {file.name} (Red Label): {e}")
        return None

    if not results:
        return None

    return pd.concat(results, ignore_index=True)


def insert_platform_currency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    After column E (5th col) insert 'YouTube' column.
    After column J (10th col) insert 'USD' column.
    These positions refer to the FINAL OUTPUT dataframe columns.
    """
    out = df.copy()

    # Insert after E => at index 5 (0-based insert position)
    pos_youtube = min(5, len(out.columns))
    out.insert(pos_youtube, "Platform", "YouTube")

    # Insert after J => at index 10 (0-based insert position)
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

# Enforce 12 files max
if uploaded_csvs and len(uploaded_csvs) > MAX_CSV_FILES:
    st.error(f"Poți încărca maxim {MAX_CSV_FILES} fișiere CSV. Ai selectat {len(uploaded_csvs)}.")
    st.stop()

if st.button("Generate Report"):
    if not uploaded_excel:
        st.error("Please upload the Asset Excel file first.")
        st.stop()

    if not uploaded_csvs:
        st.error("Please upload at least one CSV file.")
        st.stop()

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

        if not all_results:
            st.warning("No matching asset IDs found in any of the uploaded files.")
            st.stop()

        final_df = pd.concat(all_results, ignore_index=True, sort=False)

        # Insert requested constant columns
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
