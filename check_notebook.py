import json

# Load notebook
with open('AIRO_Competitor_Analysis v3 comprehensive - sentiment.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cells with 'amount='
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'amount=' in source:
            print(f"\n{'='*80}")
            print(f"Cell {i} (id: {cell.get('id', 'unknown')}):")
            print(f"{'='*80}")

            # Show all lines with amount=
            lines = cell['source']
            for j, line in enumerate(lines):
                if 'amount' in line.lower():
                    print(f"Line {j}: {line.rstrip()}")
