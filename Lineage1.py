import pandas as pd
import openpyxl

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
        
        if pd.notna(required):
            required_concepts = [req.strip() for req in required.split(",") if req.strip()
            if req.strip() and req.strip().lower() != concept.lower()]
            dependencies[concept] = required_concepts
        else:
            dependencies[concept] = []
    # all_nodes = set(dependencies.keys())   # set of uniquely referenced dependencies
    # #print (dependencies)

    # for i in range(3,mapping_df.max_row + 1) :        
    #     source_variable = mapping_df.cell(row=i, column=SourceIdVarCol).value
    #     target_variable = mapping_df.cell(row=i, column=TargetIdVarCol).value
    #     for j in range(2,variables_df.max_row + 1) :
    #         if source_variable== variables_df.cell(row=j, column=VarIdCol).value :
    #             source_dataset = variables_df.cell(row=j, column=DsNameCol).value
                
    #             if source_dataset:
    #                 all_nodes.add(source_dataset)
            
    #         if target_variable== variables_df.cell(row=j, column=VarIdCol).value :
    #             target_dataset = variables_df.cell(row=j, column=DsNameCol).value
    #             if target_dataset:
    #                 all_nodes.add(target_dataset)
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

    
    # mapping_concepts_by_level = sorted(dependencies.keys(), key=lambda x: levels.get(x, float('inf')))
    
    # for concept in mapping_concepts_by_level:
    #     for _, row in mapping_df[mapping_df['MappingConcept'] == concept].iterrows():
    #         mapping_concept = row['MappingConcept']
    #         source_variable = row['sourceVariables.id']
    #         target_variable = row['targetVariable.id']

    #         source_dataset = find_dataset(source_variable, variables_df, datasets_df)
    #         if source_dataset and levels[source_dataset] == -1:
    #             levels[source_dataset] = levels.get(mapping_concept, -1) - 1

    # for concept in mapping_concepts_by_level:
    #     for _, row in mapping_df[mapping_df['MappingConcept'] == concept].iterrows():
    #         mapping_concept = row['MappingConcept']
    #         source_variable = row['sourceVariables.id']
    #         target_variable = row['targetVariable.id']

    #         target_dataset = find_dataset(target_variable, variables_df, datasets_df)
    #         if target_dataset and levels[target_dataset] == -1:
    #             levels[target_dataset] = levels.get(mapping_concept, 0) + 1

    # for node in levels:
    #     if levels[node] == -1:
    #         levels[node] = 0

    return levels

mapping_file = 'LineageExcel/Mapping_MIMIC_DM_v04_lim.xlsx'
mapping_df, variables_df, datasets_df = load_data(mapping_file)
levels = define_levels(mapping_df, variables_df, datasets_df)

print (levels)
