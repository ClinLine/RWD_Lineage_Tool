import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tkinter import messagebox, Tk

# Load data
df1 = pd.read_csv('/Users/vedika/Desktop/cdisc/csv files/Mapping_new_dsBased_PD.xlsx - MappingConcepts.csv', header=1)
codemap = pd.read_csv('/Users/vedika/Desktop/cdisc/csv files/Mapping_new_dsBased_PD.xlsx - CodeMap.csv', header=1)
codes = pd.read_csv('/Users/vedika/Desktop/cdisc/csv files/Mapping_new_dsBased_PD.xlsx - Codes.csv', header=0)

def handle_transformation_click(source_var=None, target_var=None, transformation_type=None):
    conditions = True
   # if mapping_concept_id:
    #    conditions &= (df1.iloc[:, 0] == mapping_concept_id)
    if source_var:
        conditions &= (df1['sourceVariables.id'] == source_var)
    if target_var:
        conditions &= (df1['targetVariable.id'] == target_var)
    if transformation_type:
        conditions &= (df1['transformationType'] == transformation_type)

    matches = df1[conditions]
    if matches.empty:
        print("No matching row found.")
        return

    row = matches.iloc[0]
    transformation_type = row['transformationType'].strip().upper()

    if transformation_type == "NONE":
        root = Tk()
        root.withdraw()
        messagebox.showinfo("No Transformation", "This transformation is of type 'none'. No details available.")
        root.destroy()

    elif transformation_type == "RECODE":
        codemap_ids = row['Codemaps.Id']
        if pd.isna(codemap_ids):
            print("No codemap IDs found.")
            return

        codemap_id_list = [cid.strip() for cid in str(codemap_ids).split(',')]
        edges = []
        notes = {}
        node_ids = set()

        for cid in codemap_id_list:
            submap = codemap[codemap['id'].astype(str).str.strip() == cid]
            for _, crow in submap.iterrows():
                from_id = crow['from.id']
                to_id = crow['to.id']
                note = crow.get('note', '')
                edges.append((from_id, to_id))
                node_ids.update([from_id, to_id])
                if pd.notna(note):
                    notes[(from_id, to_id)] = note

        label_map = codes.set_index('id')['code'].to_dict()
        node_labels = {nid: label_map.get(nid, nid) for nid in node_ids}

        G = nx.DiGraph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)

        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=2000,
                node_color='lightblue', font_weight='bold', arrows=True)

        for (src, tgt), note in notes.items():
            mid_x = (pos[src][0] + pos[tgt][0]) / 2
            mid_y = (pos[src][1] + pos[tgt][1]) / 2 - 0.05
            plt.text(mid_x, mid_y, note, fontsize=9, ha='center', color='red', fontstyle='italic')

        plt.title(f"Recoding Graph: {source_var or ''} → {target_var or ''}")
        plt.show()

    elif transformation_type in ["TRANSFORM", "ASSIGN", "UNIQUE_NUMBER"]:
        text = row.get('customTransformationText', '')
        code = row.get('customTransformationCode', '')
        if pd.isna(text) and pd.isna(code):
            root = Tk()
            root.withdraw()
            messagebox.showinfo("Details", "No details provided for this transformation.")
            root.destroy()
        else:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_axis_off()
            ax.text(0.5, 0.7, f"Custom Text:\n{text}", ha='center', fontsize=10)
            ax.text(0.5, 0.4, f"Custom Code:\n{code}", ha='center', fontsize=10)
            plt.title(f"{transformation_type} Details")
            plt.show()

if __name__ == "__main__":
    # Example usage
    handle_transformation_click(mapping_concept_id="MC_2", source_var="VAR34", target_var="VAR19", transformation_type="RECODE")
