import pandas as pd
import os
import networkx as nx
from helper_stuff import regions_to_color
from find_ref_helper import clean_text2
from query_DB import query_DB

def count_books(cell):
    if pd.isna(cell):
        return 0
    else:
        books = [book.strip() for book in cell.split(';') if book.strip()]
        return len(books)

def calc_network_regions_satistics(bio_file, word_count_file, DB_final_edges_EN, DB_nodes_path):
    # Load the bio file and clean the 'NAME' column
    df_bio = pd.read_excel(bio_file, engine='openpyxl')
    df_bio['NAME'] = df_bio['NAME'].apply(lambda x: clean_text2(x)[0])

    # Load the word count file and clean the 'text_author' column
    df_word_count = pd.read_excel(word_count_file, engine='openpyxl')
    df_word_count['text_author'] = df_word_count['text_author'].apply(lambda x: clean_text2(x)[0])

    # Merge the two dataframes on the Hebrew name
    merged_df = pd.merge(df_bio, df_word_count, left_on='NAME', right_on='text_author')
    
    # Calculate total word count for each region
    total_word_count_by_region = merged_df.groupby('REGION')['Word Count'].sum()
    total_word_count_by_region = total_word_count_by_region.drop("Middle East", axis=0)
    total_word_count_by_region.name = 'Number of tokens'
    total_word_count_by_region = total_word_count_by_region.to_frame()
    
    print("==============Number of tokens per region===============")
    print(total_word_count_by_region)
    print("==============Total number of tokens===============")
    print(total_word_count_by_region.sum())
    
    network_regions_satistics_df = total_word_count_by_region

    # Filter out rows where the 'Note' column has values 'Book' or 'Unknown'
    filtered_df = df_bio[~df_bio['Note'].isin(['Book', 'Unknown'])]
    region_counts_filtered = filtered_df['REGION'].value_counts()
    region_counts_filtered = region_counts_filtered.drop("Middle East", axis=0)
    region_counts_filtered.name = 'Number of Rabies per region'
    region_counts_filtered = region_counts_filtered.to_frame()
    
    print("==============Rabies per region===============")
    print(region_counts_filtered)
    network_regions_satistics_df = pd.merge(network_regions_satistics_df, region_counts_filtered, on="REGION")

    # Filter out rows where the 'Books' column is NaN
    filtered_df_authors_only = df_bio[~df_bio['Note'].isin(['Book', 'Unknown']) & ~df_bio['Books'].isna()]
    filtered_df_authors_only = filtered_df_authors_only['REGION'].value_counts()
    filtered_df_authors_only = filtered_df_authors_only.drop("Middle East", axis=0)
    filtered_df_authors_only.name = 'Number of authors per region'
    filtered_df_authors_only = filtered_df_authors_only.to_frame()
    
    print("==============Authors per region===============")
    print(filtered_df_authors_only)
    network_regions_satistics_df = pd.merge(network_regions_satistics_df, filtered_df_authors_only, on="REGION")

    # Apply the function to the 'Books' column to get the number of books per row
    df_bio['Book_Count'] = df_bio['Books'].apply(count_books)
    books_per_region = df_bio.groupby('REGION')['Book_Count'].sum()
    books_per_region = books_per_region.drop("Middle East", axis=0)
    books_per_region.name = 'Number of books per region'
    books_per_region = books_per_region.to_frame()
    
    print("==============Books per region===============")
    print(books_per_region)
    network_regions_satistics_df = pd.merge(network_regions_satistics_df, books_per_region, on="REGION")

    # Calculate every region's out-degree
    df_regions_out_degree = pd.DataFrame(columns=["Average Regions out degree"])
    num_rows = 0
    total_len_of_out_degree_centrality = 0
    century_period_filter = [11, 12, 13, 14, 15]

    for region in regions_to_color:
        GEO_filter = [region]
        region_DB_DF = query_DB(DB_final_edges_EN, DB_nodes_path, "", GEO_filter, centuries_period_filter=century_period_filter)
        region_centrality_path = os.path.join("/Users/natibengigi/Library/Mobile Documents/com~apple~CloudDocs/Education/PHD/PHD_scripts/CODE/Main_branch_code/Article 2 results V3", "region_centrality_debug.xlsx")
        region_DB_DF.to_excel(region_centrality_path, engine='openpyxl')

        num_rows += len(region_DB_DF.index)
        G = nx.from_pandas_edgelist(region_DB_DF, 'Source', 'Target', 'Weight', create_using=nx.DiGraph())
        out_degree_centrality = nx.out_degree_centrality(G)
        filtered_out_degree_centrality = {node: centrality for node, centrality in out_degree_centrality.items() if centrality > 0}
        df_regions_out_degree.loc[region] = sum(filtered_out_degree_centrality.values()) / len(filtered_out_degree_centrality)
        
        print(f"Region outdegree: {region}")
        print(df_regions_out_degree.loc[region])
        
        filtered_out_degree_centrality_len = len(filtered_out_degree_centrality)
        print(f"Out degree centrality for {region}: {filtered_out_degree_centrality_len}")
        total_len_of_out_degree_centrality += filtered_out_degree_centrality_len

    df_regions_out_degree.index.name = 'REGION'
    df_regions_out_degree = df_regions_out_degree.drop("Middle East", axis=0)
    
    print("==============Out degree per region===============")
    print(df_regions_out_degree)
    network_regions_satistics_df = pd.merge(network_regions_satistics_df, df_regions_out_degree, on="REGION")

    # Calculate overall network out-degree centrality
    full_network_DF = query_DB(DB_final_edges_EN, DB_nodes_path, "", "", centuries_period_filter=century_period_filter)
    print(f"Total aggregate num of entries (for all regions) = {num_rows}")
    print(f"Num of entries for the entire network = {len(full_network_DF)}")
    G = nx.from_pandas_edgelist(full_network_DF, 'Source', 'Target', 'Weight', create_using=nx.DiGraph())
    
    full_network_out_degree_centrality = nx.out_degree_centrality(G)
    filtered_out_degree_centrality = {node: centrality for node, centrality in full_network_out_degree_centrality.items() if centrality > 0}
    average_full_network_out_degree_centrality = sum(filtered_out_degree_centrality.values()) / len(filtered_out_degree_centrality)
    
    print(f"Average full network out-degree centrality: {average_full_network_out_degree_centrality}")
    print(f"Total aggregate of out-degree centrality = {total_len_of_out_degree_centrality}")
    print(f"Filtered out-degree centrality = {len(filtered_out_degree_centrality)}")

    full_network_in_degree_centrality = nx.in_degree_centrality(G)
    filtered_in_degree_centrality = {node: centrality for node, centrality in full_network_in_degree_centrality.items() if centrality > 0}
    average_full_network_in_degree_centrality = sum(filtered_in_degree_centrality.values()) / len(filtered_in_degree_centrality)
    
    print(f"Average full network in-degree centrality: {average_full_network_in_degree_centrality}")

    full_network_degree_centrality = nx.degree_centrality(G)
    filtered_full_degree_centrality = {node: centrality for node, centrality in full_network_degree_centrality.items() if centrality > 0}
    average_full_network_degree_centrality = sum(filtered_full_degree_centrality.values()) / len(filtered_full_degree_centrality)
    
    print(f"Average full network degree centrality: {average_full_network_degree_centrality}")

    average_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print(f"Average Degree: {average_degree}")

    edge_density = nx.density(G)
    print(f"Edge Density: {edge_density}")

    average_in_degree = sum(dict(G.in_degree()).values()) / G.number_of_nodes()
    print(f"Average In-Degree: {average_in_degree}")

    print(network_regions_satistics_df)
    return network_regions_satistics_df, average_full_network_out_degree_centrality


