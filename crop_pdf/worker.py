# -------------------------------------------------------------
# worker.py
# This script processes ONLY ONE PDF file and then exits.
# -------------------------------------------------------------
import sys
import fitz

def get_o2sat_crop_coordinates(page_width, page_height):
    """Get fixed coordinates for O2Sat image based on PDF dimensions"""
    left = int(page_width * 0.046)
    right = int(page_width * 0.325)
    top = int(page_height * 0.5185)
    bottom = int(page_height * 0.721)
    return (left, top, right, bottom)

def crop_single_pdf(pdf_path_str, output_path_str):
    """
    Opens, crops, and saves a single PDF file.
    """
    try:
        with fitz.open(pdf_path_str) as doc:
            page = doc[0]
            page_rect = page.rect
            
            # Get the crop box for the O2Sat image
            clip_rect = get_o2sat_crop_coordinates(page_rect.width, page_rect.height)
            
            # Get the pixmap (image) of the cropped area
            pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=clip_rect, alpha=False)
            
            # Save the image
            pix.save(output_path_str)
            print(f"[WORKER OK] Successfully cropped '{pdf_path_str}' to '{output_path_str}'")
            
    except Exception as e:
        print(f"[WORKER ERROR] Failed to process {pdf_path_str}: {e}")

if __name__ == "__main__":
    # The main script will provide these three arguments
    if len(sys.argv) == 3:
        pdf_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        crop_single_pdf(pdf_file_path, output_file_path)
    else:
        print("[WORKER USAGE] python worker.py <path_to_pdf> <path_to_output_png>")
