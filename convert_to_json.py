import json

# Relative path to the .ipynb file
ipynb_file = 'Notebooks/exploration.ipynb'

# Read the .ipynb file
with open(ipynb_file, 'r', encoding='utf-8') as f:
    notebook_content = json.load(f)

# Save the JSON to a new file
output_json_file = 'Notebooks/exploration.json'
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=4)

print(f"Notebook has been converted to JSON and saved as {output_json_file}")