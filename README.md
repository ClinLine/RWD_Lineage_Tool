# Real-World Data Lineage Use Case and Visualization Tool


## Visualization tool 
The Real World Data Lineage Tool aims to vizualize the lineage of data from source to submission ready datasets. 
  Based on a generalized model structure and including all details necessary for the data transformations the tool vizualizes the mapping information on different levels:
  - Each selection, mapping and/or transformation step resulting in a new data file, is represented as a (pink) mapping node in the lineage graph.
      - All datasets and/or alternative sources (like json, xml etc) necessary for this step are presented in blue as input for the mapping node
      - Reuse of datasets for multiple purposes is presented by more outgoing arrows from the same dataset.
  - Double clicking on a mapping node results in an overview of the details for that step      
    - Input datasets are shown in dark blue
    - Potential corresponding joins between these datasets which may limit the output (inner, left/right outer joins) are presented as a diamond. 
      The details of this join are shown when double clicking on the diamond representing the link between two datasets.
    - Variables of the input dataset as well as the output dataset are shown in light blue squares linked to the corresponding dataset. 
      The transformation between the input and output dataset is represented by the middle squares. Double clikcing on these squares will give details for this transformation.

## Parkinson Use Case
Mappings based on a Parkinson Mock Protocol will be added to the Github RWE_Lineage_Tool space. This includes 
  - The Mock Protocol
  - Mapping from MIMIC FHIR formatted datasource to an OMOP-like structure and subsequently to SDTM submissable datasets which can be utilized in the visualisation tool
  - Define.xml like documentation for the mappings

## License

This project is using the [MIT](http://www.opensource.org/licenses/MIT "The MIT License | Open Source Initiative") license for code and scripts ![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
