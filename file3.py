import pandas as pd
import matplotlib.pyplot as plt

def display_link_details(link_name):
    """Finds and displays the row for the given link name."""
    df = pd.read_csv('/Users/vedika/Desktop/cdisc/csv files/Mapping_new_dsBased_PD.xlsx - DatasetLink.csv')
    matching_row = df[df['link'] == link_name]

    if matching_row.empty:
        print(f"No data found for link: {link_name}")
        return
    
    # Convert row data to a dictionary (excluding 'link' column)
    row_data = matching_row.iloc[0].drop("link").to_dict()
    
    # Display the data in a new pop-up window
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_axis_off()

    # Title
    ax.text(0.5, 1, f"Details for: {link_name}", fontsize=12, fontweight="bold", ha="center")

    # Display each field in a centered format
    for i, (key, value) in enumerate(row_data.items()):
        ax.text(0.5, 0.8 - (i * 0.15), f"{key}: {value}", fontsize=10, ha="center")

    plt.show()

# Sample DataFrame for testing
data = {
    "link": ["example1", "example2", "example3"],
    "Description": ["First link details", "Second link details", "Third link details"],
    "Category": ["A", "B", "C"]
}
df = pd.DataFrame(data)

# Test function
display_link_details("example1")