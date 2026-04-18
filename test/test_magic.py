import subprocess
import glob
import os

pdf_files = glob.glob('data/resumes/*.pdf')
if len(pdf_files) > 1:
    p = pdf_files[1] # The real one
    print(f'Testing {p}')
    out_dir = 'data/resumes/out'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    res = subprocess.run(f'magic-pdf -p "{p}" -o "{out_dir}" -m auto', shell=True, capture_output=True, text=True, errors='replace')
    print('STDOUT:', res.stdout)
    print('STDERR:', res.stderr)
    
    # Check directory contents
    print("Contents formed:")
    for dp, _, fns in os.walk(out_dir):
        for f in fns:
            print(os.path.join(dp, f))
