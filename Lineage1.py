import pandas as pd
import openpyxl
import networkx as nx  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def load_data(workbook):
    wb = openpyxl.load_workbook(workbook)
    mapping_df = wb['MappingConcepts']
    variables_df = wb['Variables']
    datasets_df = wb['Dataset']    
    return mapping_df, variables_df, datasets_df

def Column_names(sheet, headerRow):
    ColNames = []
    ColNames.append("") #add empty value to align names with sheet column names
    for i in range(1, sheet.max_column + 1):
        ColName = sheet.cell(row=headerRow, column=i).value
        ColNames.append(ColName)
    #print(ColNames)
    return ColNames    

def define_levels(mapping_df, variables_df, datasets_df):
    levels = {}
    dependencies = {}    
    map_col = Column_names(mapping_df,2)
    var_col = Column_names(variables_df,1)
    ds_col = Column_names(variables_df,1)
    for x in map_col:
        if x == 'Required': ReqCol = map_col.index(x)  
        if x == 'sourceVariables.id': SourceIdVarCol = map_col.index(x)
        if x == 'targetVariable.id': TargetIdVarCol = map_col.index(x)
    for x in var_col:
        if x == 'id': VarIdCol = var_col.index(x) 
        if x == 'parent.name': DsNameCol = var_col.index(x)
    for x in ds_col:
        if x == 'id': DsIdCol = ds_col.index(x)

    for i in range(3,mapping_df.max_row + 1) :
        concept = mapping_df.cell(row=i, column=1).value # id will be on first column
        required = mapping_df.cell(row=i, column=ReqCol).value
        
        if pd.notna(required) and required != '':
            required_concepts = [req.strip() for req in required.split(",") if req.strip()
            if req.strip() and req.strip().lower() != concept.lower()]
            dependencies[concept] = required_concepts
        else:
            if concept != '' and concept != None:
                dependencies[concept] = []
    
    levels = {node: -1 for node in dependencies}
    
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
    print("Mapping concepts by level:", mapping_concepts_by_level)
    
    for i in range(3,mapping_df.max_row + 1) :   # add in and output datasets     
        mapping_concept = mapping_df.cell(row=i, column=1).value # id will be on first column
        source_variable = mapping_df.cell(row=i, column=SourceIdVarCol).value
        target_variable = mapping_df.cell(row=i, column=TargetIdVarCol).value
        for j in range(2,variables_df.max_row + 1) :
            if source_variable== variables_df.cell(row=j, column=VarIdCol).value :
                source_dataset = variables_df.cell(row=j, column=DsNameCol).value
                if source_dataset != '' and source_dataset != None and levels.get(source_dataset, 999) > levels.get(mapping_concept, -1) - 1:
                    levels[source_dataset] = levels.get(mapping_concept, -1) - 1

            if target_variable== variables_df.cell(row=j, column=VarIdCol).value :
                target_dataset = variables_df.cell(row=j, column=DsNameCol).value
                if target_dataset != '' and target_dataset != None and levels.get(target_dataset, 999) > levels.get(mapping_concept, -1) - 1:
                    levels[target_dataset] = levels.get(mapping_concept, -1) - 1

    return levels

def build_graph(mapping_df, variables_df, datasets_df):
    G = nx.DiGraph()  # Create a directed graph
    levels = define_levels(mapping_df, variables_df, datasets_df)
    edges = []

    for level in levels:
        if levels[level] / 2 == int(levels[level] / 2):
            G.add_node(level, type="Dataset", color="lightgreen")
        else:
            G.add_node(level, type="MappingConcept", color="lightblue")
    
    # for _, row in mapping_df.iterrows():
    #     mapping_concept = row['MappingConcept']
    #     source_variable = row['sourceVariables.id']
    #     target_variable = row['targetVariable.id']

    #     source_dataset = find_dataset(source_variable, variables_df, datasets_df)
    #     target_dataset = find_dataset(target_variable, variables_df, datasets_df)

    #     # Add mapping concept nodes with light blue color
    #     G.add_node(mapping_concept, type="MappingConcept", color="lightblue")

    #     if source_dataset:
    #         # Add source dataset nodes with light green color
    #         G.add_node(source_dataset, type="Dataset", color="lightgreen")
    #         edges.append((source_dataset, mapping_concept))
        
    #     if target_dataset:
    #         # Add target dataset nodes with light green color
    #         G.add_node(target_dataset, type="Dataset", color="lightgreen")
    #         edges.append((mapping_concept, target_dataset))
        
    #     required = row.get('Required', None)
    #     if pd.notna(required):
    #         required_concepts = [req.strip() for req in required.split(",") if req.strip()]
    #         for required_concept in required_concepts:
    #             if required_concept != mapping_concept:
    #                 G.add_node(required_concept, type="MappingConcept", color="lightblue")
    #                 edges.append((required_concept, mapping_concept))

    # Add edges to the graph
    #for edge in edges:
    # Remove edges where both nodes are MappingConcepts
    # edges = [edge for edge in edges if not (G.nodes[edge[0]].get('type') == 'MappingConcept' and G.nodes[edge[1]].get('type') == 'MappingConcept')]
    # G.add_edges_from(edges)
    
    

    return G, levels


def draw_graph_with_buttons(G, levels):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)
    
    level_nodes = {}
    for node, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)
    print("level_nodes", level_nodes)
    
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

mapping_file = 'LineageExcel/Mapping_MIMIC_DM_v04_lim.xlsx'
# mapping_df, variables_df, datasets_df = load_data(mapping_file)
# G, levels = build_graph(mapping_df, variables_df, datasets_df)

# draw_graph_with_buttons(G, levels)
np.random.seed(0)
x = np.random.randn(1000)
y = np.random.randn(1000)
colors = np.random.randint(10, 101, size=1000)
sizes = np.random.randint(10, 101, size=1000)

# Scatter plot with multiple customizations
plt.scatter(x, y, c=colors, cmap="viridis", s=sizes, marker='o', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Matplotlib')
plt.show()

