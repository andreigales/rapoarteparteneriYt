import streamlit as st
import pandas as pd
from pandas.errors import ParserError
import zipfile
import rarfile
import io
import os
import gc
import shutil
import tempfile

# ---------------------------------------------------------------------------
# RAR backend selection
# ---------------------------------------------------------------------------
# `rarfile` is a pure-Python wrapper that shells out to an external binary
# (unrar / unar / bsdtar / 7z). On Streamlit Community Cloud `unrar` is in
# the non-free Debian repo and cannot be installed, so we install `unar`,
# `libarchive-tools` (bsdtar) and `p7zip-full` (7z) via packages.txt and
# let rarfile pick whichever is available (it prefers unar, then bsdtar,
# then 7z by default — unar is the most reliable for RAR5).
def _configure_rar_backend():
    for tool_name, attr in (
        ("unar", "UNAR_TOOL"),
        ("unrar", "UNRAR_TOOL"),
        ("bsdtar", "BSDTAR_TOOL"),
        ("7z", "SEVENZIP_TOOL"),
        ("7zz", "SEVENZIP2_TOOL"),
    ):
        path = shutil.which(tool_name)
        if path:
            setattr(rarfile, attr, path)
            return tool_name, path
    return None, None

RAR_BACKEND_NAME, RAR_BACKEND_PATH = _configure_rar_backend()

# =========================
# Settings (stability)
# =========================
MAX_CSV_FILES = 12
CHUNK_SIZE = 200_000  # reduce (e.g. 50_000) if you still hit RAM issues
COPY_BUF_SIZE = 1024 * 1024  # 1 MiB stream copy buffer

# =========================
# Helper: robust chunk reader (accepts a path or file-like)
# =========================
def iter_chunks_with_fallback(source, header=None, skiprows=0, chunksize=CHUNK_SIZE):
    """
    Iterates CSV chunks robustly.
    - source can be a filesystem path (str) or a file-like object.
    - First try: engine='c', sep=',' with on_bad_lines='skip'
    - If parsing fails (including mid-iteration): restart and fallback to
      engine='python', sep=None (auto-detect)
    """
    def _rewind():
        try:
            source.seek(0)
        except Exception:
            pass

    def make_reader(engine, sep):
        _rewind()
        return pd.read_csv(
            source,
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

def process_standard_csv(source, display_name, asset_ids):
    """Standard report: chunk-read, filter by asset_id column index 3, keep desired cols."""
    DESIRED_COLS = [1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 26, 27]
    asset_col_idx = 3

    results = []
    try:
        for chunk in iter_chunks_with_fallback(source, header=None, skiprows=0):
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

def process_redlabel_csv(source, display_name, asset_ids):
    """Red Label report: read header names (row 2), then chunk-read data from row 3 onward."""
    DESIRED_COLS = [1, 2, 3, 4, 6, 9, 11, 12, 13, 17, 26]
    asset_col_idx = 3

    # Read header names (row 2 => skiprows=1)
    header_names = []
    try:
        try:
            source.seek(0)
        except Exception:
            pass
        hdr_df = pd.read_csv(
            source,
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
        for chunk in iter_chunks_with_fallback(source, header=None, skiprows=2):
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

    pos_youtube = min(5, len(out.columns))
    out.insert(pos_youtube, "Platform", "YouTube")

    pos_usd = min(10, len(out.columns))
    out.insert(pos_usd, "Currency", "USD")

    return out

# =========================
# Archive handling (ZIP + RAR) — STREAMING TO DISK
# =========================
# Loading every CSV into a BytesIO at once blows the 1 GiB RAM limit on
# Streamlit Community Cloud. Instead, we stream each entry straight from
# the archive to a temporary directory on disk and process from disk.
def _is_csv_candidate(name: str) -> bool:
    """Return True if the archive entry looks like a real CSV file."""
    base = os.path.basename(name)
    if not base:
        return False
    if base.startswith("__MACOSX") or "/__MACOSX" in name:
        return False
    if base.startswith("."):
        return False
    return base.lower().endswith(".csv")

def _unique_path(dest_dir: str, base: str) -> str:
    """Return a path in dest_dir that doesn't yet exist, suffixing if needed."""
    out_path = os.path.join(dest_dir, base)
    counter = 0
    while os.path.exists(out_path):
        counter += 1
        name, ext = os.path.splitext(base)
        out_path = os.path.join(dest_dir, f"{name}__{counter}{ext}")
    return out_path

def extract_zip_to_dir(zip_source, dest_dir: str):
    """Stream CSVs from a ZIP (file-like) to dest_dir. Returns [(display_name, path), ...]."""
    out = []
    with zipfile.ZipFile(zip_source) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if not _is_csv_candidate(info.filename):
                continue
            base = os.path.basename(info.filename)
            out_path = _unique_path(dest_dir, base)
            with z.open(info) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=COPY_BUF_SIZE)
            out.append((base, out_path))
    return out

def extract_rar_to_dir(rar_source, dest_dir: str):
    """Stream CSVs from a RAR (file-like) to dest_dir. Returns [(display_name, path), ...]."""
    if not RAR_BACKEND_PATH:
        raise RuntimeError(
            "RAR backend not available on the server (no unar/unrar/bsdtar/7z on PATH). "
            "Check packages.txt."
        )
    out = []
    with rarfile.RarFile(rar_source) as r:
        for info in r.infolist():
            if info.is_dir():
                continue
            if not _is_csv_candidate(info.filename):
                continue
            base = os.path.basename(info.filename)
            out_path = _unique_path(dest_dir, base)
            with r.open(info) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=COPY_BUF_SIZE)
            out.append((base, out_path))
    return out

def extract_archive_to_dir(archive_file, dest_dir: str):
    """Dispatch by extension. archive_file is a Streamlit UploadedFile."""
    name = archive_file.name.lower()
    # Streamlit's UploadedFile is a BytesIO-like with .seek(0).
    try:
        archive_file.seek(0)
    except Exception:
        pass
    if name.endswith(".zip"):
        return extract_zip_to_dir(archive_file, dest_dir)
    elif name.endswith(".rar"):
        return extract_rar_to_dir(archive_file, dest_dir)
    else:
        return []

# ==========
# Main UI
# ==========
st.set_page_config(page_title="YouTube Asset Reporter", layout="wide")

st.title("YouTube Asset Report Generator")
st.markdown(
    "Poti incarca fie pana la 12 fisiere CSV, fie una sau mai multe arhive ZIP/RAR care contin CSV-uri."
)

if RAR_BACKEND_NAME:
    st.caption(f"RAR backend: {RAR_BACKEND_NAME} ({RAR_BACKEND_PATH})")
else:
    st.caption("RAR backend: NOT AVAILABLE - only ZIP and direct CSV will work.")

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

uploaded_archives = st.file_uploader(
    "Variant B: Upload ZIP/RAR archive(s) containing CSV files",
    type=["zip", "rar"],
    accept_multiple_files=True,
    key="archive_multi",
)

# Enforce 12 max for multi-CSV selection
if uploaded_csvs and len(uploaded_csvs) > MAX_CSV_FILES:
    st.error(f"Poti incarca maxim {MAX_CSV_FILES} fisiere CSV. Ai selectat {len(uploaded_csvs)}.")
    st.stop()


# Generate
if st.button("Generate Report"):
    if not uploaded_excel:
        st.error("Please upload the Asset Excel file first.")
        st.stop()

    if not uploaded_archives and not uploaded_csvs:
        st.error("Te rog incarca fie CSV-uri, fie un ZIP/RAR cu CSV-uri.")
        st.stop()

    with st.spinner("Processing files..."):
        asset_ids = load_asset_ids(uploaded_excel)
        st.success(f"Loaded {len(asset_ids)} asset IDs.")

        if not asset_ids:
            st.error("Lista de asset IDs este goala (sau nu a putut fi citita).")
            st.stop()

        all_results = []

        # Everything below runs inside a TemporaryDirectory so we never keep
        # extracted CSV bytes resident in memory.
        with tempfile.TemporaryDirectory(prefix="rapoarte_") as tmpdir:
            csv_items = []  # list of (display_name, path)

            # Archives path
            if uploaded_archives:
                for archive_file in uploaded_archives:
                    try:
                        extracted = extract_archive_to_dir(archive_file, tmpdir)
                        if not extracted:
                            st.warning(
                                f"Arhiva '{archive_file.name}' nu contine niciun fisier .csv."
                            )
                            continue
                        csv_items.extend(extracted)
                    except Exception as e:
                        st.error(f"Nu am putut citi arhiva '{archive_file.name}': {e}")
                        st.stop()
                    finally:
                        # Force release any internal buffers in Streamlit's UploadedFile.
                        try:
                            archive_file.seek(0)
                        except Exception:
                            pass
                        gc.collect()

                if not csv_items:
                    st.warning("No CSV files found in any of the uploaded archives.")
                    st.stop()

                if len(csv_items) > MAX_CSV_FILES:
                    st.error(
                        f"Total CSV-uri din toate arhivele: {len(csv_items)}. "
                        f"Maxim permis: {MAX_CSV_FILES}."
                    )
                    st.stop()

                st.info(
                    f"Am gasit {len(csv_items)} fisiere CSV in "
                    f"{len(uploaded_archives)} arhiva(e). Procesare in flux..."
                )

            # Direct CSV upload path — also persist to disk so we use the same
            # streaming code path and let the original UploadedFile be released.
            elif uploaded_csvs:
                for f in uploaded_csvs:
                    base = os.path.basename(f.name)
                    out_path = _unique_path(tmpdir, base)
                    try:
                        f.seek(0)
                    except Exception:
                        pass
                    with open(out_path, "wb") as dst:
                        shutil.copyfileobj(f, dst, length=COPY_BUF_SIZE)
                    csv_items.append((base, out_path))

            progress_bar = st.progress(0)

            for i, (display_name, csv_path) in enumerate(csv_items):
                if report_type == "Standard Report":
                    res = process_standard_csv(csv_path, display_name, asset_ids)
                else:
                    res = process_redlabel_csv(csv_path, display_name, asset_ids)

                if res is not None and not res.empty:
                    all_results.append(res)

                # Drop the file from disk as soon as we're done with it.
                try:
                    os.remove(csv_path)
                except Exception:
                    pass

                # Encourage GC between files.
                gc.collect()
                progress_bar.progress((i + 1) / len(csv_items))

        # tmpdir auto-cleanup happens here (TemporaryDirectory context).

        if not all_results:
            st.warning("No matching asset IDs found in any of the uploaded files.")
            st.stop()

        final_df = pd.concat(all_results, ignore_index=True, sort=False)
        # Free the per-file frames now that we've concatenated.
        all_results.clear()
        gc.collect()

        final_df = insert_platform_currency_columns(final_df)

        st.write(f"### Found {len(final_df)} matching rows")
        st.dataframe(final_df.head(50))

        csv_bytes = final_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download Final Report CSV",
            data=csv_bytes,
            file_name=f"report_{report_type.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )
