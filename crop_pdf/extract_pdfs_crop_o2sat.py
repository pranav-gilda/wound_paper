# -------------------------------------------------------------
# extract_pdfs_crop_o2sat.py (Main Controller - FINAL VERSION)
# -------------------------------------------------------------
import re, pathlib, fitz, pandas as pd, logging, sys, subprocess

ROOT_PDF_DIR = pathlib.Path("sensor project images")
IMG_DIR      = pathlib.Path("extracted_imgs"); IMG_DIR.mkdir(exist_ok=True, parents=True)
OUT_CSV      = "pdf_master_table.csv"
LOG_FILE     = "pdf_extract.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, "w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# ---------- regex patterns ----------
re_dt   = re.compile(r"Report Date:\s*([0-9/]{8,10}\s+[0-9:]{6,8}(?:\s*[APMapm]{2})?)")
re_temp = re.compile(r"Skin Temperature:\s*([0-9.]+)\s*°?C")

def extract_visit_number(pdf_path):
    filename = pdf_path.stem.lower()
    visit_match = re.search(r'v\.?(\d+)', filename)
    if visit_match:
        return f"V{visit_match.group(1)}"
    return "V0"

rows = []
processed_count = 0

# Get all PDFs, skip "none" folders, and sort them
all_pdfs = []
for pdf in ROOT_PDF_DIR.rglob("*.pdf"):
    if "none" in pdf.parent.name.lower():
        logging.info(f"Skipping folder with 'none': {pdf.parent.name}")
        continue
    all_pdfs.append(pdf)

all_pdfs.sort(key=lambda x: (int(re.search(r'#?(\d+)', x.parent.name).group(1)) if re.search(r'#?(\d+)', x.parent.name) else 999, x.name))

logging.info(f"Found {len(all_pdfs)} PDF files to process")

for pdf_path in all_pdfs:
    try:
        # --- Part 1: Extract metadata from the PDF ---
        with fitz.open(pdf_path) as doc:
            txt = doc[0].get_text()
            sid = re.search(r"#?(\d{1,4})", pdf_path.parent.name) or re.search(r"#?(\d{1,4})", pdf_path.stem)
            if not sid:
                logging.warning(f"[WARN] no study_id in {pdf_path}")
                continue
            
            study_id = int(sid.group(1))
            visit = extract_visit_number(pdf_path)
            dt_match = re_dt.search(txt)
            report_dt = pd.to_datetime(dt_match.group(1), errors="coerce") if dt_match else pd.NaT
            temp_match = re_temp.search(txt)
            temp_c = float(temp_match.group(1)) if temp_match else None

        # --- Part 2: Delegate the cropping to the isolated worker script ---
        img_name = f"{study_id}_{visit}_o2sat.png"
        out_png = IMG_DIR / img_name

        logging.info(f"--- Delegating crop for {pdf_path.name} to worker ---")
        
        command = [sys.executable, "worker.py", str(pdf_path), str(out_png)]
        
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        logging.info(f"[OK] Worker finished for {img_name}")
        
        rows.append(
            dict(
                study_id=study_id,
                visit_tag=visit,
                report_datetime=report_dt,
                skin_temp_c=temp_c,
                img_o2sat=str(out_png),
            )
        )
        processed_count += 1
        print(f"--- Processed {processed_count} / {len(all_pdfs)} files ---")


    except subprocess.CalledProcessError as e:
        logging.error(f"[ERROR] Worker script failed for {pdf_path.name}.")
        logging.error(f"Worker output:\n{e.stdout}\n{e.stderr}")
    except Exception as e:
        logging.error(f"[ERROR] Main script failed on {pdf_path.name}: {e}")

# Once the loop is finished, write the CSV
if rows:
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    logging.info(f"=== extracted {len(rows)} rows to {OUT_CSV} ===")
logging.info(f"=== processed {processed_count} files total ===")
# -------------------------------------------------------------