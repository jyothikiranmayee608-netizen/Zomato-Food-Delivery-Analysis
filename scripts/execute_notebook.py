"""Execute the notebook using nbclient (no jupyter command required)."""
import nbformat
from nbclient import NotebookClient

NB_PATH = "notebooks/zomato_delivery_analysis.ipynb"

nb = nbformat.read(NB_PATH, as_version=4)
client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": "notebooks/"}}
)
print("Executing notebook...")
client.execute()

# Write back with outputs
nbformat.write(nb, NB_PATH)
print(f"Notebook executed and saved: {NB_PATH}")
print(f"Total cells: {len(nb.cells)}")
errors = [c for c in nb.cells if c.cell_type == "code" and
          any(o.get("output_type") == "error" for o in c.get("outputs", []))]
if errors:
    print(f"WARNING: {len(errors)} cells had errors")
    for c in errors:
        for o in c["outputs"]:
            if o.get("output_type") == "error":
                print(f"  {o.get('ename')}: {o.get('evalue')}")
else:
    print("All cells executed without errors.")
