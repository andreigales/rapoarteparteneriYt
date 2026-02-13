import streamlit as st
import pandas as pd
from pandas.errors import ParserError
import zipfile
import io
import os

# =========================
# Settings (stability)
# =========================
MAX_CSV_FILES = 12
CHUNK_SIZE = 200_000  # reduce (e.g. 50_000) if you still hit RAM issues


# =========================
# Helper: robust chunk reader
# =========================
def iter_chunks_with_fallback(file_like, header=None, skiprows=0, chunksize=CHUNK_SIZE):
    """
    Iterates CSV chunks robustly.
    - First try: engine='c', sep=',' with on_bad_lines='skip'
    - If parsing fails (including mid-iteration): restart and fallback to engine='python', sep=None (auto-detect)
    """
    def make_reader(engine, sep):
        try:
            file_like.seek(0)
        except Exception:
            pass
        return pd.read_csv(
            file_like,
            header=header,
            skiprows=skiprows,
            dtype=str,
            low_memory=False,
            chunksize=chunksize,
            engine=engine,
            sep=sep,
            on_bad_lines="skip",
        )

    # First attempt
    try:
        reader = make_reader(engine="c", sep=",")
        for chunk in reader:
            yield chunk
        return
    except ParserError:
        pass
    except Exception:
        # For any other issue, fallback as well
        pass

    # Fallback attempt (restart)
    reader2 = make_reader(engine="python", sep=None)
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


def process_standard_csv(file_like, display_name, asset_ids):
    """Standard report: chunk-read, filter by asset_id column index 3, keep desired cols."""
    DESIRED_COLS = [1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 26, 27]
    asset_col_idx = 3

    results = []
    try:
        for chunk in iter_chunks_with_fallback(file_like, header=None, skiprows=0):
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
            valid_cols = [i for i in DESIRED_COLS if i < cols]

            sub = filtered[valid_cols].copy()
            sub["source_file"] = display_name
            results.append(sub)

    except Exception as e:
        st.warning(f"Error reading {display_name}: {e}")
        return None

    if not results:
        return None
    return pd.concat(results, ignore_index=True)


def process_redlabel_csv(file_like, display_name, asset_ids):
    """Red Label report: read header names (row 2), then chunk-read data from row 3 onward."""
    DESIRED_COLS = [1, 2, 3, 4, 6, 9, 11, 12, 13, 17, 26]
    asset_col_idx = 3

    # Read header names (row 2 => skiprows=1)
    header_names = []
    try:
        file_like.seek(0)
        hdr_df = pd.read_csv(
            file_like,
            nrows=0,
            skiprows=1,
            engine="python",
            sep=None,
            on_bad_lines="skip",
        )
        header_names = list(hdr_df.columns)
    except Exception as e:
        st.warning(f"Could not read header for {display_name} (Red Label): {e}")
        header_names = []

    results = []
    try:
        for chunk in iter_chunks_with_fallback(file_like, header=None, skiprows=2):
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

            new_cols = []
            for i in valid_indices:
                if i < len(header_names):
                    new_cols.append(header_names[i])
                else:
                    new_cols.append(f"col_{i}")
            sub.columns = new_cols

            sub["source_file"] = display_name
            results.append(sub)

    except Exception as e:
        st.warning(f"Error processing {display_name} (Red Label): {e}")
        return None

    if not results:
        return None
    return pd.concat(results, ignore_index=True)


def insert_platform_currency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    After column E (5th col) insert a column filled with 'YouTube'.
    After column J (10th col) insert a column filled with 'USD'.
    Positions refer to FINAL OUTPUT dataframe columns (A,B,C,...).
    """
    out = df.copy()

    # After E => insert at index 5
    pos_youtube = min(5, len(out.columns))
    out.insert(pos_youtube, "Platform", "YouTube")

    # After J => insert at index 10
    pos_usd = min(10, len(out.columns))
    out.insert(pos_usd, "Currency", "USD")

    return out


# =========================
# ZIP handling
# =========================
def extract_csv_filelikes_from_zip(zip_bytes: bytes):
    """
    Returns list of tuples: (display_name, file_like)
    where file_like is a BytesIO ready for pandas.
    """
    csv_items = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            # Skip folders / hidden files
            if info.is_dir():
                continue
            name = info.filename
            base = os.path.basename(name)
            if not base:
                continue
            if base.startswith("__MACOSX") or "/__MACOSX" in name:
                continue
            if base.startswith("."):
                continue
            if base.lower().endswith(".csv"):
                data = z.read(info)
                csv_items.append((base, io.BytesIO(data)))
    return csv_items


# ==========
# Main UI
# ==========
st.set_page_config(page_title="YouTube Asset Reporter", layout="wide")

st.title("📊 YouTube Asset Report Generator")
st.markdown(
    "Poți încărca fie până la 12 fișiere CSV, fie un singur ZIP care conține până la 12 CSV-uri."
)

report_type = st.sidebar.radio("Select Report Type", ["Standard Report", "Red Label Report"])

st.subheader("1. Upload Inputs")
uploaded_excel = st.file_uploader("Upload 'assets soundfeed.xlsx'", type=["xlsx", "xls"])

st.markdown("### Alege una dintre variante:")
uploaded_csvs = st.file_uploader(
    "Variant A: Upload CSV Files (multiple)",
    type=["csv"],
    accept_multiple_files=True,
    key="csv_multi",
)

uploaded_zip = st.file_uploader(
    "Variant B: Upload a ZIP containing CSV files",
    type=["zip"],
    accept_multiple_files=False,
    key="zip_single",
)

# Enforce 12 max for multi-CSV selection
if uploaded_csvs and len(uploaded_csvs) > MAX_CSV_FILES:
    st.error(f"Poți încărca maxim {MAX_CSV_FILES} fișiere CSV. Ai selectat {len(uploaded_csvs)}.")
    st.stop()

# Prepare inputs list: (display_name, file_like)
input_csv_items = []

# If ZIP provided, use ZIP (recommended for the 'max 6' file-picker issue)
if uploaded_zip is not None:
    try:
        zip_bytes = uploaded_zip.getvalue()
        extracted = extract_csv_filelikes_from_zip(zip_bytes)
        if not extracted:
            st.error("ZIP-ul nu conține niciun fișier .csv.")
            st.stop()

        if len(extracted) > MAX_CSV_FILES:
            st.error(f"ZIP-ul conține {len(extracted)} CSV-uri. Maxim permis: {MAX_CSV_FILES}.")
            st.stop()

        input_csv_items = extracted
        st.info(f"Am găsit {len(input_csv_items)} fișiere CSV în ZIP.")
    except Exception as e:
        st.error(f"Nu am putut citi ZIP-ul: {e}")
        st.stop()

# Else use direct CSV uploads
elif uploaded_csvs:
    # Each UploadedFile is already file-like
    input_csv_items = [(f.name, f) for f in uploaded_csvs]

# Generate
if st.button("Generate Report"):
    if not uploaded_excel:
        st.error("Please upload the Asset Excel file first.")
        st.stop()

    if not input_csv_items:
        st.error("Te rog încarcă fie CSV-uri, fie un ZIP cu CSV-uri.")
        st.stop()

    with st.spinner("Processing files..."):
        asset_ids = load_asset_ids(uploaded_excel)
        st.success(f"Loaded {len(asset_ids)} asset IDs.")

        if not asset_ids:
            st.error("Lista de asset IDs este goală (sau nu a putut fi citită).")
            st.stop()

        all_results = []
        progress_bar = st.progress(0)

        for i, (display_name, file_like) in enumerate(input_csv_items):
            # Ensure pointer at start for each file
            try:
                file_like.seek(0)
            except Exception:
                pass

            if report_type == "Standard Report":
                res = process_standard_csv(file_like, display_name, asset_ids)
            else:
                res = process_redlabel_csv(file_like, display_name, asset_ids)

            if res is not None and not res.empty:
                all_results.append(res)

            progress_bar.progress((i + 1) / len(input_csv_items))

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
