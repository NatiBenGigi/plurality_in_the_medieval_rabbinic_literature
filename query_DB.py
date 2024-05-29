import pandas as pd

def query_DB(DB_final_edges_EN, nodes_DB, output_path, GEO_filter, years_period_filter=None, centuries_period_filter=None):
    # Check if both filters were supplied
    if years_period_filter and centuries_period_filter:
        raise ValueError("Cannot supply both years_period_filter and centuries_period_filter. Please supply only one.")

    # Read the edges and nodes files
    df_edges = pd.read_excel(DB_final_edges_EN)
    df_nodes = pd.read_excel(nodes_DB)

    # Filter based on GEO_filter
    if GEO_filter:
        df_nodes = df_nodes[df_nodes['REGION'].isin(GEO_filter)]

    # Filter based on years_period_filter
    if years_period_filter:
        if len(years_period_filter) == 1:
            df_nodes = df_nodes[df_nodes['DATE OF BIRTH'] >= years_period_filter[0]]
        else:
            df_nodes = df_nodes[df_nodes['DATE OF BIRTH'].between(years_period_filter[0], years_period_filter[1])]

    # Filter based on centuries_period_filter
    if centuries_period_filter:
        df_nodes = df_nodes[df_nodes['CENTURY'].isin(centuries_period_filter)]

    # Filter edges based on filtered nodes
    df_edges = df_edges[df_edges['Source'].isin(df_nodes['Label'])]

    if output_path:
        # Save the filtered edges to the output file
        df_edges.to_excel(output_path, index=False)
        print(f"Filtered edges saved to: {output_path}")

    return df_edges

