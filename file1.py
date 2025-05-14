import pandas as pd
import networkx as nx  
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from link1 import run_graph  # Import the function from link1.py

def load_data(mapping_file, variables_file, datasets_file):
    mapping_df = mapping_file, header=1
    variables_df = pd.read_csv(variables_file)
    datasets_df = pd.read_csv(datasets_file)
    
    mapping_df.columns = mapping_df.columns.str.strip()
    mapping_df.rename(columns={mapping_df.columns[0]: 'MappingConcept'}, inplace=True)
   
    mapping_df['sourceVariables.id'] = mapping_df['sourceVariables.id'].fillna(method='bfill')
    
    empty_mapping_concept_rows = mapping_df['MappingConcept'].isna() | (mapping_df['MappingConcept'].str.strip() == '')
    
    for idx in range(1, len(mapping_df)):
        if empty_mapping_concept_rows[idx]:
            if pd.isna(mapping_df.loc[idx-1, 'MappingConcept']) or mapping_df.loc[idx-1, 'MappingConcept'].strip() == '':
                mapping_df.loc[idx, 'MappingConcept'] = None
            else:
                mapping_df.loc[idx, 'MappingConcept'] = mapping_df.loc[idx-1, 'MappingConcept']
        
        if empty_mapping_concept_rows[idx]:
            if pd.isna(mapping_df.loc[idx-1, 'Required']) or mapping_df.loc[idx-1, 'Required'].strip() == '':
                mapping_df.loc[idx, 'Required'] = None
            else:
                mapping_df.loc[idx, 'Required'] = mapping_df.loc[idx-1, 'Required']
    
    mapping_df = mapping_df[mapping_df['MappingConcept'].notna() & (mapping_df['MappingConcept'] != '')]
    mapping_df['MappingConcept'] = mapping_df['MappingConcept'].replace(r'^\s*$', '', regex=True).str.strip()
    mapping_df = mapping_df[mapping_df['MappingConcept'].notna() & (mapping_df['MappingConcept'] != '')]

    return mapping_df, variables_df, datasets_df

def find_dataset(variable_id, variables_df, datasets_df):
    parent_id = variables_df.loc[variables_df['id'] == variable_id, 'parent.id'].values
    if len(parent_id) > 0:
        dataset = datasets_df.loc[datasets_df['id'] == parent_id[0]]
        if not dataset.empty:
            return dataset['label'].values[0]
    return None

def define_levels(mapping_df, variables_df, datasets_df):
    levels = {}
    dependencies = {}

    for _, row in mapping_df.iterrows():
        concept = row['MappingConcept']
        required = row.get('Required', None)

        if pd.notna(required):
            required_concepts = [req.strip() for req in required.split(",") if req.strip()
            if req.strip() and req.strip().lower() != concept.lower()
                                 ]


            dependencies[concept] = required_concepts
        else:
            dependencies[concept] = []

    all_nodes = set(dependencies.keys())
    for _, row in mapping_df.iterrows():
        source_variable = row['sourceVariables.id']
        target_variable = row['targetVariable.id']

        source_dataset = find_dataset(source_variable, variables_df, datasets_df)
        target_dataset = find_dataset(target_variable, variables_df, datasets_df)

        if source_dataset:
            all_nodes.add(source_dataset)
        if target_dataset:
            all_nodes.add(target_dataset)

    levels = {node: -1 for node in all_nodes}

    for concept, required_concepts in dependencies.items():
        if not required_concepts:
            levels[concept] = 1

    while True:
        updated = False
        for concept, required_concepts in dependencies.items():
            if levels[concept] == -1:
                if all(levels[req] != -1 for req in required_concepts):
                    levels[concept] = max(levels[req] for req in required_concepts) + 2
                    updated = True

        if not updated:
            break

    mapping_concepts_by_level = sorted(dependencies.keys(), key=lambda x: levels.get(x, float('inf')))
    
    for concept in mapping_concepts_by_level:
        for _, row in mapping_df[mapping_df['MappingConcept'] == concept].iterrows():
            mapping_concept = row['MappingConcept']
            source_variable = row['sourceVariables.id']
            target_variable = row['targetVariable.id']

            source_dataset = find_dataset(source_variable, variables_df, datasets_df)
            if source_dataset and levels[source_dataset] == -1:
                levels[source_dataset] = levels.get(mapping_concept, -1) - 1

    for concept in mapping_concepts_by_level:
        for _, row in mapping_df[mapping_df['MappingConcept'] == concept].iterrows():
            mapping_concept = row['MappingConcept']
            source_variable = row['sourceVariables.id']
            target_variable = row['targetVariable.id']

            target_dataset = find_dataset(target_variable, variables_df, datasets_df)
            if target_dataset and levels[target_dataset] == -1:
                levels[target_dataset] = levels.get(mapping_concept, 0) + 1

    for node in levels:
        if levels[node] == -1:
            levels[node] = 0

    return levels

def build_graph(mapping_df, variables_df, datasets_df):
    G = nx.DiGraph()  # Create a directed graph
    levels = define_levels(mapping_df, variables_df, datasets_df)
    edges = []
    
    for _, row in mapping_df.iterrows():
        mapping_concept = row['MappingConcept']
        source_variable = row['sourceVariables.id']
        target_variable = row['targetVariable.id']

        source_dataset = find_dataset(source_variable, variables_df, datasets_df)
        target_dataset = find_dataset(target_variable, variables_df, datasets_df)

        # Add mapping concept nodes with light blue color
        G.add_node(mapping_concept, type="MappingConcept", color="lightblue")

        if source_dataset:
            # Add source dataset nodes with light green color
            G.add_node(source_dataset, type="Dataset", color="lightgreen")
            edges.append((source_dataset, mapping_concept))
        
        if target_dataset:
            # Add target dataset nodes with light green color
            G.add_node(target_dataset, type="Dataset", color="lightgreen")
            edges.append((mapping_concept, target_dataset))
        
        required = row.get('Required', None)
        if pd.notna(required):
            required_concepts = [req.strip() for req in required.split(",") if req.strip()]
            for required_concept in required_concepts:
                if required_concept != mapping_concept:
                    G.add_node(required_concept, type="MappingConcept", color="lightblue")
                    edges.append((required_concept, mapping_concept))

    # Add edges to the graph
    #for edge in edges:
    # Remove edges where both nodes are MappingConcepts
    edges = [edge for edge in edges if not (G.nodes[edge[0]].get('type') == 'MappingConcept' and G.nodes[edge[1]].get('type') == 'MappingConcept')]
    G.add_edges_from(edges)
    
    

    return G, levels
    
def on_click(event, G, pos):
    """Handles node clicks and triggers the detailed graph for MappingConcepts."""
    print(f"Click detected at ({event.xdata}, {event.ydata})")  # Debugging line
    
    if event.xdata is None or event.ydata is None:
        return  # Ignore clicks outside the graph

    for node, (x, y) in pos.items():
        print(f"Checking node '{node}' at ({x}, {y})")  # Debugging line
        
        if abs(event.xdata - x) < 0.3 and abs(event.ydata - y) < 0.3:
            node_type = G.nodes[node].get('type', None)
            
            if node_type == "MappingConcept":
                print(f"Clicked Mapping Concept: {node}")  # Debugging line
                run_graph(node)  # Call detailed graph function in link1.py
            break

def draw_graph_with_buttons(G, levels):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)
    
    level_nodes = {}
    for node, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)
    
    pos = {}
    max_nodes_per_level = max(len(nodes) for nodes in level_nodes.values())
    y_spacing = max(1, max_nodes_per_level // 2)
    
    for level, nodes_at_level in sorted(level_nodes.items()):
        for i, node in enumerate(nodes_at_level):
            pos[node] = (level, -i * y_spacing)
    
    mapping_concepts = [node for node in G.nodes if G.nodes[node].get('type') == "MappingConcept"]
    datasets = [node for node in G.nodes if G.nodes[node].get('type') == "Dataset"]
    node_colors = [G.nodes[node].get('color', 'gray') for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=mapping_concepts, node_color='#ffcc99', node_size=2500, node_shape="s", alpha=0.8, ax=ax)  # squares
    nx.draw_networkx_nodes(G, pos, nodelist=datasets, node_color='lightblue', node_size=2500, node_shape="s", alpha=0.8, ax=ax)  # Squares



    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrowstyle='-|>', arrowsize=10, connectionstyle="arc3,rad=0.2", ax=ax)
    

    
    ax.set_title("Mapping Concepts and Datasets (Click a Node)")
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, G, pos))
    plt.show()

def main():
    mapping_file = '/Users/vedika/Desktop/cdisc/new files/Mapping_MIMIC_DM_v03.xlsx - MappingConcepts.csv'
    variables_file = '/Users/vedika/Desktop/cdisc/new files/Mapping_MIMIC_DM_v03.xlsx - Variables.csv'
    datasets_file = '/Users/vedika/Desktop/cdisc/new files/Mapping_MIMIC_DM_v03.xlsx - Dataset.csv'

    mapping_df, variables_df, datasets_df = load_data(mapping_file, variables_file, datasets_file)
    G, levels = build_graph(mapping_df, variables_df, datasets_df)
    draw_graph_with_buttons(G, levels)

if __name__ == "__main__":
    main()
