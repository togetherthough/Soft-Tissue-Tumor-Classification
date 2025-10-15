import json
from pathlib import Path

nb_path = Path("Experiment1-Benchmarks.ipynb")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

updated = False

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        
        # Find cells that call run_multi_tabpfn or run_multi_localpfn
        if ('run_multi_tabpfn(' in source or 'run_multi_localpfn(' in source) and 'checkpoint=' not in source:
            lines = cell['source']
            new_lines = []
            
            for line in lines:
                new_lines.append(line)
                # Add checkpoint parameter after outputs_base_dir
                if 'outputs_base_dir=' in line and ',' in line:
                    # Add checkpoint parameter on next line
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + 'checkpoint=checkpoint_path,\n')
                    updated = True
            
            if updated:
                cell['source'] = new_lines
                print(f"✅ Added checkpoint=checkpoint_path to function call")
                updated = False  # Reset for next cell

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("💾 Saved Experiment1-Benchmarks.ipynb with checkpoint parameter")
