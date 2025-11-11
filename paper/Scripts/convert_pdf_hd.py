#!/usr/bin/env python3
"""
Convert PDF to ultra high resolution PNG using matplotlib
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import PyPDF2
from PIL import Image
import fitz  # PyMuPDF
from pathlib import Path

figures_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
pdf_path = figures_dir / "joint_numbering_diagram.pdf"
output_path = figures_dir / "joint_numbering_diagram_ultra_hd.png"

try:
    # Try using PyMuPDF for highest quality
    pdf_document = fitz.open(str(pdf_path))
    page = pdf_document[0]

    # Render at very high resolution (4x normal)
    mat = fitz.Matrix(4, 4)  # 4x zoom for ultra high quality
    pix = page.get_pixmap(matrix=mat)

    # Save the high-res image
    pix.save(str(output_path))
    print(f" Ultra HD conversion successful using PyMuPDF")

except ImportError:
    print("PyMuPDF not available, trying alternative method...")

    # Alternative: Use matplotlib to render PDF
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib import rcParams

    # Set high DPI for rendering
    rcParams['figure.dpi'] = 600

    # Create a figure and load the PDF as an image
    # This is a workaround - display the PDF page
    from PIL import Image
    import subprocess

    # Use sips with maximum quality settings
    subprocess.run([
        'sips', '-s', 'format', 'png',
        '-s', 'formatOptions', 'best',
        '--resampleHeightWidthMax', '3000',
        str(pdf_path), '--out', str(output_path)
    ])
    print(f" Ultra HD conversion using sips with max quality")