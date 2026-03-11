import os

def clean_file(path):
    print(f"Cleaning {path}")
    with open(path, 'rb') as f:
        data = f.read().decode('utf-8', 'ignore')
    
    # Simple ASCII map
    replacements = {
        '\u2014': '-',
        '\u2013': '-',
        '\u201c': '"',
        '\u201d': '"',
        '\u2018': "'",
        '\u2019': "'",
        '\u2713': '[OK]',
        '\u2554': '+',
        '\u2557': '+',
        '\u255a': '+',
        '\u255d': '+',
        '\u2551': '|',
        '\u2550': '-',
        '\u2026': '...',
        '\u2501': '-',
        '\u2500': '-',
    }
    
    for k, v in replacements.items():
        data = data.replace(k, v)
    
    # Strip any remaining non-ascii
    data = "".join(i if ord(i) < 128 else "" for i in data)
    
    with open(path, 'w', encoding='ascii') as f:
        f.write(data)

base_dir = r"e:\MLOps\LLM Failure Evaluation Engine\python\rlrgf"
for root, dirs, files in os.walk(base_dir):
    for name in files:
        if name.endswith('.py'):
            clean_file(os.path.join(root, name))
