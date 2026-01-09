import json

# Load notebook
with open('AIRO_Competitor_Analysis v3 comprehensive - sentiment.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix cells with 'amount=' parameter
changes_made = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'amount=' in source or 'limit=' in source:
            print(f"\nCell {i} (id: {cell.get('id', 'unknown')}):")
            print(f"  Has 'amount=': {'amount=' in source}")
            print(f"  Has 'limit=': {'limit=' in source}")

            # Show relevant lines
            lines = cell['source']
            for j, line in enumerate(lines):
                if 'amount=' in line or 'limit=' in line:
                    print(f"  Line {j}: {line[:80]}")

            # Make replacement
            new_source = []
            for line in lines:
                new_line = line.replace('amount=num_filings', 'limit=num_filings')
                new_line = new_line.replace('amount=5', 'limit=5')
                if new_line != line:
                    changes_made += 1
                    print(f"  Changed: {line.strip()[:60]} -> {new_line.strip()[:60]}")
                new_source.append(new_line)

            cell['source'] = new_source

print(f"\n\nTotal changes made: {changes_made}")

# Save notebook
if changes_made > 0:
    with open('AIRO_Competitor_Analysis v3 comprehensive - sentiment.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("✓ Notebook saved with fixes")
else:
    print("No changes needed - already correct")
