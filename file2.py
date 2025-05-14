import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from link import display_link_details
from transformation import handle_transformation_click


def load_data(mapping_file, variables_file, datasets_file, link_file):
    """Loads and processes data from CSV files."""
    mapping_df = pd.read_csv(mapping_file, header=1)
    variables_df = pd.read_csv(variables_file)
    datasets_df = pd.read_csv(datasets_file)
    link_df = pd.read_csv(link_file)  # Load the link file

    # Clean Mapping Data
    mapping_df.columns = mapping_df.columns.str.strip()
    mapping_df.rename(columns={mapping_df.columns[0]: 'MappingConceptID'}, inplace=True)
    mapping_df['sourceVariables.id'] = mapping_df['sourceVariables.id'].bfill()  # Fix for FutureWarning

    if 'transformationType' in mapping_df.columns:
        mapping_df['transformationType'] = mapping_df['transformationType'].replace('UNIQUE_NUMBER', 'UNIQUE NUMBER')


    # Forward fill MappingConceptID and Required fields
    empty_mapping_concept_rows = mapping_df['MappingConceptID'].isna() | (mapping_df['MappingConceptID'].str.strip() == '')
    for idx in range(1, len(mapping_df)):
        if empty_mapping_concept_rows[idx]:
            mapping_df.loc[idx, 'MappingConceptID'] = mapping_df.loc[idx-1, 'MappingConceptID'] if not pd.isna(mapping_df.loc[idx-1, 'MappingConceptID']) else None
            mapping_df.loc[idx, 'Required'] = mapping_df.loc[idx-1, 'Required'] if not pd.isna(mapping_df.loc[idx-1, 'Required']) else None

    mapping_df = mapping_df[mapping_df['MappingConceptID'].notna() & (mapping_df['MappingConceptID'] != '')]
    mapping_df['MappingConceptID'] = mapping_df['MappingConceptID'].replace(r'^\s*$', '', regex=True).str.strip()

    return mapping_df, variables_df, datasets_df, link_df

def find_dataset(variable_id, variables_df, datasets_df):
    """Finds the dataset associated with a variable ID."""
    parent_id = variables_df.loc[variables_df['id'] == variable_id, 'parent.id'].values
    if len(parent_id) > 0:
        dataset = datasets_df.loc[datasets_df['id'] == parent_id[0]]
        if not dataset.empty:
            return dataset['label'].values[0]
    return None

def find_datalabel(name,datasets_df):
    
    dataset = datasets_df.loc[datasets_df['name'] == name]
    if not dataset.empty:
        return dataset['label'].values[0]
    return None

def build_graph(mapping_df, variables_df, datasets_df, link_df, mapping_concept_id):
    """Constructs a directed graph based on a single MappingConcept ID."""
    import networkx as nx
    import pandas as pd

    G = nx.DiGraph()
    dataset_nodes = {}  # Track added dataset nodes to avoid duplication

    # Create lookup maps
    parent_id_to_label = dict(zip(datasets_df['id'], datasets_df['label']))
    parent_name_to_label= dict(zip(datasets_df['name'], datasets_df['label']))
    variable_id_to_label = dict(zip(variables_df['id'], variables_df['label']))
    variable_id_to_dataset = dict(zip(variables_df['id'], variables_df['parent.id']))

    # Filter for relevant rows
    filtered_df = mapping_df[mapping_df['MappingConceptID'] == mapping_concept_id]
    if filtered_df.empty:
        print(f"No data found for MappingConcept ID: {mapping_concept_id}")
        return None

    source_datasets = set()

    for _, row in filtered_df.iterrows():
        source_var_id = row['sourceVariables.id']
        target_var_id = row['targetVariable.id'] if pd.notna(row['targetVariable.id']) else None
        transformation_type = row['transformationType'] if pd.notna(row['transformationType']) else 'NONE'

        # Handle source variable label
        source_label = variable_id_to_label.get(source_var_id)
        if source_label is None:
            continue  # Skip if source label is missing
        source_var_name = f"{source_label}_source"

        # Handle target variable label (may be missing)
                # Handle target variable label (may be missing)
        target_var_name = None
        valid_target = False
        if target_var_id in variable_id_to_label:
            target_label = variable_id_to_label[target_var_id]
            if pd.notna(target_label):
                target_var_name = f"{target_label}_target"
                valid_target = True


        # Define dataset labels
        source_dataset_id = variable_id_to_dataset.get(source_var_id)
        target_dataset_id = variable_id_to_dataset.get(target_var_id) if target_var_id else None
        source_dataset_label = parent_id_to_label.get(source_dataset_id, source_dataset_id)
        target_dataset_label = parent_id_to_label.get(target_dataset_id, target_dataset_id) if target_dataset_id else None

        # Mark source datasets for link connections
        if source_dataset_label:
            source_datasets.add(source_dataset_label)

        # Add dataset nodes (once)
        if source_dataset_label and source_dataset_label not in dataset_nodes:
            G.add_node(source_dataset_label, type='dataset')
            dataset_nodes[source_dataset_label] = True
        if target_dataset_label and target_dataset_label not in dataset_nodes:
            G.add_node(target_dataset_label, type='dataset')
            dataset_nodes[target_dataset_label] = True

        # Add source variable node + edge
        G.add_node(source_var_name, type='svariable')
        if source_dataset_label:
            G.add_edge(source_dataset_label, source_var_name)

        # Add transformation node + edge
        transformation_node_id = f"{transformation_type}_{source_var_id}_{target_var_id if target_var_id else 'NA'}"
        G.add_node(transformation_node_id, type='transformation', label=transformation_type)
        G.add_edge(source_var_name, transformation_node_id)

        if target_var_id and target_var_id in variable_id_to_label:
            target_label = variable_id_to_label[target_var_id]
            if pd.notna(target_label):
                target_var_name = f"{target_label}_target"
                G.add_node(target_var_name, type='tvariable')
                G.add_edge(transformation_node_id, target_var_name)

                # Also link to dataset if target dataset is valid
                target_dataset_id = variable_id_to_dataset.get(target_var_id)
                target_dataset_label = parent_id_to_label.get(target_dataset_id, target_dataset_id)
                if target_dataset_label:
                    G.add_node(target_dataset_label, type='dataset')  # just in case
                    G.add_edge(target_var_name, target_dataset_label)


        # Add edge to target dataset
        if target_dataset_label and target_var_name:
            G.add_edge(target_var_name, target_dataset_label)

    # Add dataset link nodes
    for _, link_row in link_df.iterrows():
        dataset1_id = link_row['dataset1']
        dataset2_id = link_row['dataset2']
        linkid = link_row['id']

        dataset1_label = parent_name_to_label.get(dataset1_id, str(dataset1_id))
        dataset2_label = parent_name_to_label.get(dataset2_id, str(dataset2_id))

        if dataset1_label in dataset_nodes and dataset2_label in dataset_nodes:
            if linkid not in G.nodes:
                G.add_node(linkid, type='link', label=linkid)
            G.add_edge(dataset1_label, linkid)
            G.add_edge(dataset2_label, linkid)
        print (link_df)
        print("Checking link:", dataset1_label, dataset2_label)
        print("Available dataset nodes:", dataset_nodes)
        print(f"Row: {dataset1_id}, {dataset2_id}, linkid: {linkid}")
        print(f"Labels: {dataset1_label}, {dataset2_label}")
        print(f"Exists in dataset_nodes? {dataset1_label in dataset_nodes}, {dataset2_label in dataset_nodes}")
        print (dataset_nodes)
        print(parent_name_to_label)


    return G



def on_click(event, G, pos):
    print(f"Click detected at ({event.xdata}, {event.ydata})")  # Debugging line
    
    if event.xdata is None or event.ydata is None:
        return  # Ignore clicks outside the graph

    for node, (x, y) in pos.items():
        #print(f"Checking node '{node}' at ({x}, {y})")  # Debugging line
        
        if abs(event.xdata - x) < 0.3 and abs(event.ydata - y) < 0.3:
            node_type = G.nodes[node].get('type', None)
            
            if node_type == "link":
                print(f"Clicked Mapping Concept: {node}") 
                display_link_details(node)  # Pass node or link_node_id to this function
                #ax.text(x, y, label, fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))
            
            elif node_type == 'transformation':
                parts = node.split("_")  
                transformation_type = parts[0]  # First part
                source_var = parts[1]    # Second part
                target_var = parts[2]
                handle_transformation_click(transformation_type, source_var, target_var)
                print (transformation_type, source_var, target_var)
            
        else:
                    label = G.nodes[node].get('label', str(node))
            # ax.text(x, y, label, fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))
    plt.draw()
        #break


def draw_graph(G, mapping_concept_id, mapping_df, variables_df, datasets_df):
    """Draws a clean, grouped, left-to-right graph for a given MappingConceptID."""

    if G is None:
        return

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    variable_to_label = dict(zip(variables_df['id'], variables_df['label']))
    variable_to_dataset = dict(zip(variables_df['id'], variables_df['parent.id']))
    dataset_id_to_label = dict(zip(datasets_df['id'], datasets_df['label']))

    x_offset = {
        'dataset_source': -2, 'variable_source': -1, 'transformation': 0,
        'variable_target': 1, 'dataset_target': 2, 'link': -3
    }

    pos = {}
    y_cursor = 0
    source_dataset_y_positions = {}

    mc_df = mapping_df[mapping_df['MappingConceptID'] == mapping_concept_id].copy()
    mc_df['sourceDataset'] = mc_df['sourceVariables.id'].map(variable_to_dataset)
    mc_df = mc_df.sort_values(by=['sourceDataset', 'sourceVariables.id'])

    for _, row in mc_df.iterrows():
        svar = row['sourceVariables.id']
        tvar = row['targetVariable.id'] if pd.notna(row['targetVariable.id']) else None
        trans_type = row['transformationType'] if pd.notna(row['transformationType']) else 'NONE'

        source_dataset_id = variable_to_dataset.get(svar)
        source_dataset_label = dataset_id_to_label.get(source_dataset_id, source_dataset_id)

        target_dataset_id = variable_to_dataset.get(tvar) if tvar else None
        target_dataset_label = dataset_id_to_label.get(target_dataset_id, target_dataset_id) if target_dataset_id else None

        svar_label = variable_to_label.get(svar)
        if pd.isna(svar_label):
            continue
        svar_node = f"{svar_label}_source"

        tvar_node = None
        if tvar:
            tvar_label = variable_to_label.get(tvar)
            if pd.notna(tvar_label):
                tvar_node = f"{tvar_label}_target"

        transformation_node = f"{trans_type}_{svar}_{tvar if tvar else 'NA'}"

        if source_dataset_label:
            pos[source_dataset_label] = (x_offset['dataset_source'], -y_cursor)
            source_dataset_y_positions[source_dataset_label] = -y_cursor
        pos[svar_node] = (x_offset['variable_source'], -y_cursor)
        pos[transformation_node] = (x_offset['transformation'], -y_cursor)
        if tvar_node:
            pos[tvar_node] = (x_offset['variable_target'], -y_cursor)
        if target_dataset_label:
            pos[target_dataset_label] = (x_offset['dataset_target'], -y_cursor)

        y_cursor += 2

    for node in G.nodes:
        if G.nodes[node].get('type') == 'link':
            linked_datasets = [n for n in G.predecessors(node) if G.nodes[n].get('type') == 'dataset']
            y_vals = [source_dataset_y_positions.get(ds) for ds in linked_datasets if ds in source_dataset_y_positions]
            if y_vals:
                pos[node] = (x_offset['link'], sum(y_vals) / len(y_vals))

    fig, ax = plt.subplots(figsize=(25, 8))

    safe_edges = [(u, v) for u, v in G.edges() if u in pos and v in pos]

    def side_offset(node, direction):
        ntype = G.nodes[node].get('type')
        if ntype in ['svariable', 'tvariable']:
            return 0.35 if direction == 'out' else -0.35
        elif ntype == 'transformation':
            return 0.25 if direction == 'out' else -0.25
        elif ntype == 'dataset':
            return 0.5 if direction == 'out' else -0.5
        else:
            return 0

    for u, v in safe_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        u_type = G.nodes[u].get('type')
        v_type = G.nodes[v].get('type')

        # Apply offsets only if not a dataset
        offset_u = 0 if u_type == 'dataset' else side_offset(u, 'out')
        offset_v = 0 if v_type == 'dataset' else side_offset(v, 'in')

        start = (x1 + offset_u, y1)
        end = (x2 + offset_v, y2)

        ax.annotate("",
            xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
        )


    def draw_nodes(nodelist, color, shape, size, alpha):
        safe_nodes = [n for n in nodelist if n in pos]
        nx.draw_networkx_nodes(G, pos, nodelist=safe_nodes, node_color=color,
                               node_shape=shape, node_size=size, alpha=alpha, ax=ax)

    mapping_concepts = [n for n, d in G.nodes(data=True) if d['type'] == 'mapping']
    datasets = [n for n, d in G.nodes(data=True) if d['type'] == 'dataset']
    source_variables = [n for n, d in G.nodes(data=True) if d['type'] == 'svariable']
    target_variables = [n for n, d in G.nodes(data=True) if d['type'] == 'tvariable']
    transformations = [n for n, d in G.nodes(data=True) if d['type'] == 'transformation']
    links = [n for n, d in G.nodes(data=True) if d['type'] == 'link']

    draw_nodes(mapping_concepts, '#ffcc99', 'o', 2500, 0.8)
    draw_nodes(datasets, 'darkblue', 'o', 4000, 0.8)
    draw_nodes(source_variables, '#0047AB', 's', 1000, 0)
    draw_nodes(target_variables, '#0047AB', 's', 1000, 0)
    draw_nodes(transformations, 'coral', 's', 1800, 0)
    draw_nodes(links, 'darkgreen', 'd', 1800, 0.8)

    for node in source_variables + target_variables:
        if node in pos:
            x, y = pos[node]
            ax.add_patch(patches.Rectangle((x - 0.35, y - 0.65), 0.7, 1.3, linewidth=1, edgecolor='black', facecolor='#5ec2da', alpha=0.8))
    for node in transformations:
        if node in pos:
            x, y = pos[node]
            ax.add_patch(patches.Rectangle((x - 0.25, y - 0.65), 0.5, 1.3, linewidth=1, edgecolor='black', facecolor='coral', alpha=0.8))

    node_labels = {n: G.nodes[n].get('label', str(n)) for n in G.nodes}
    font_color = 'white'
    safe_labels = {n: lbl for n, lbl in node_labels.items() if n in pos and not pd.isna(n)}
    nx.draw_networkx_labels(G, pos, labels=safe_labels, font_size=6, font_weight='bold', font_color=font_color, ax=ax)

    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, G, pos))
    plt.title(f"Graph for Mapping Concept ID: {mapping_concept_id}")
    plt.axis('off')
    plt.show()



    

    def draw_nodes(nodelist, color, shape, size, alpha):
        safe_nodes = [n for n in nodelist if n in pos]
        nx.draw_networkx_nodes(G, pos, nodelist=safe_nodes, node_color=color,
                            node_shape=shape, node_size=size, alpha=alpha, ax=ax)

    mapping_concepts = [n for n, d in G.nodes(data=True) if d['type'] == 'mapping']
    datasets = [n for n, d in G.nodes(data=True) if d['type'] == 'dataset']
    source_variables = [n for n, d in G.nodes(data=True) if d['type'] == 'svariable']
    target_variables = [n for n, d in G.nodes(data=True) if d['type'] == 'tvariable']
    transformations = [n for n, d in G.nodes(data=True) if d['type'] == 'transformation']
    links = [n for n, d in G.nodes(data=True) if d['type'] == 'link']

    draw_nodes(mapping_concepts, '#ffcc99', 'o', 2500, 0.8)
    draw_nodes(datasets, 'darkblue', 'o', 4000, 0.8)
    draw_nodes(source_variables, '#0047AB', 's', 1000, 0)
    draw_nodes(target_variables, '#0047AB', 's', 1000, 0)
    draw_nodes(transformations, 'coral', 's', 1800, 0)
    draw_nodes(links, 'darkgreen', 'd', 1800, 0.8)

    # Rectangles behind variable/transform blocks
    for node in source_variables + target_variables:
        if node in pos:
            x, y = pos[node]
            ax.add_patch(patches.Rectangle((x - 0.35, y - 0.65), 0.7, 1.3, linewidth=1, edgecolor='black', facecolor='#5ec2da', alpha=0.8))
    for node in transformations:
        if node in pos:
            x, y = pos[node]
            ax.add_patch(patches.Rectangle((x - 0.25, y - 0.65), 0.5, 1.3, linewidth=1, edgecolor='black', facecolor='coral', alpha=0.8))

    # Labels
    node_labels = {n: G.nodes[n].get('label', str(n)) for n in G.nodes}
    for node in G.nodes:
        node_type = G.nodes[node].get('type', '')
        font_color = 'white' if node_type in ['svariable', 'tvariable', 'transformation'] else 'white'
    safe_labels = {n: lbl for n, lbl in node_labels.items() if n in pos and not pd.isna(n)}
    nx.draw_networkx_labels(G, pos, labels=safe_labels, font_size=6, font_weight='bold', font_color=font_color, ax=ax)

    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, G, pos))
    plt.title(f"Graph for Mapping Concept ID: {mapping_concept_id}")
    plt.show()




def run_graph(mapping_concept_id):
    """Runs the graph generation process for a specified MappingConcept ID."""
    # File paths (Replace with actual paths)
    mapping_file = '/Users/vedika/Desktop/cdisc/new files/Mapping_MIMIC_DM_v03.xlsx - MappingConcepts.csv'
    variables_file = '/Users/vedika/Desktop/cdisc/new files/Mapping_MIMIC_DM_v03.xlsx - Variables.csv'
    datasets_file = '/Users/vedika/Desktop/cdisc/new files/Mapping_MIMIC_DM_v03.xlsx - Dataset.csv'
    link_file = '/Users/vedika/Desktop/cdisc/new files/Mapping_MIMIC_DM_v03.xlsx - DatasetLink.csv'

    # Load data
    mapping_df, variables_df, datasets_df, link_df = load_data(mapping_file, variables_file, datasets_file, link_file)

    # Build graph for the given MappingConcept ID
    G = build_graph(mapping_df, variables_df, datasets_df, link_df, mapping_concept_id)

    # Draw the graph
    draw_graph(G, mapping_concept_id, mapping_df, variables_df, datasets_df)






