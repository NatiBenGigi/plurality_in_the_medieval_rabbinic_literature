import os
from references_IN_and_OUT import *
from helper_stuff import *
from plot_score_matrix import *
from general_satistics import *

def setup_paths():
    cwd = os.getcwd()
    print(cwd)

    results_folder = os.path.join(cwd, "results")
    resources_folder = os.path.join(cwd, "resources")

    paths = {
        "results_folder": results_folder,
        "resources_folder": resources_folder,
        "Biography_path": os.path.join(resources_folder, "Biography.xlsx"),
        "authors_words_number_path": os.path.join(resources_folder, "author_word_counts_and_sizes.xlsx"),
        "DB_final_edges_EN": os.path.join(resources_folder, "DB_final_edges_EN.xlsx"),
        "DB_nodes_path": os.path.join(resources_folder, "DB_nodes.xlsx"),
        "total_references_ABS": os.path.join(results_folder, 'total_references_ABS.xlsx'),
        "total_references_ratio": os.path.join(results_folder, 'total_references_ratio.xlsx'),
        "total_normalized_amount_of_ref_per_region": os.path.join(results_folder, 'total_normalized_amount_of_ref_per_region.xlsx'),
        "total_normalized_amount_of_edges_per_region": os.path.join(results_folder, 'total_normalized_amount_of_edges_per_region.xlsx'),
    }

    return paths

def main():
    paths = setup_paths()

    referred_author_ABS = os.path.join(paths["results_folder"], "referred_author_ABS.xlsx")
    _ = build_ABS_referred_author_list(paths["DB_final_edges_EN"], referred_author_ABS)

    ref_threshold = 0
    max_value_threshold = 0
    author_2_author_table = os.path.join(paths["results_folder"], "author_2_author_table.xlsx")
    build_pivot_author_2_author_table(
        referred_author_ABS,
        paths["DB_final_edges_EN"],
        paths["Biography_path"],
        ref_threshold,
        max_value_threshold,
        author_2_author_table
    )

    region_2_region_table = os.path.join(paths["results_folder"], "region_2_region_table.xlsx")
    build_pivot_region_2_region_table(author_2_author_table, region_2_region_table)

    region_2_region_table_normalized_by_word_count = os.path.join(paths["results_folder"], "region_2_region_table_normalized_by_word_count.xlsx")
    region_2_region_table_normalized_by_word_count_df, regions_authors_num_of_references_dict = build_pivot_region_2_region_table__normalized_by_word_count(
        author_2_author_table,
        paths["Biography_path"],
        paths["authors_words_number_path"],
        region_2_region_table_normalized_by_word_count
    )

    region_2_region_table_by_num_of_edges = os.path.join(paths["results_folder"], "region_2_region_table_normalized_by_by_num_of_edges.xlsx")
    region_2_region_full_total_df = build_pivot_region_2_region_table_by_num_of_edges(
        author_2_author_table,
        paths["Biography_path"],
        paths["authors_words_number_path"],
        region_2_region_table_by_num_of_edges,
        True
    )

    print(region_2_region_full_total_df)

    plt.rcParams.update({'font.size': 15})
    region_2_region_table_edges_not_normalized = os.path.join(paths["results_folder"], "region_2_region_table_edges_not_normalized.xlsx")
    sum_of_edges_per_region = build_pivot_region_2_region_table_edges_not_normalized(
        author_2_author_table,
        paths["Biography_path"],
        paths["authors_words_number_path"],
        region_2_region_table_edges_not_normalized,
        False
    )
    plot_region_2_region_table(region_2_region_table_edges_not_normalized, "Reds", "", True)

    plt.rcParams.update({'font.size': 10})
    region_OUT_vs_total_edges_dict = calculate_OUT_vs_total_edges(region_2_region_table_edges_not_normalized)
    plot_region_OUT_vs_total_edges(region_OUT_vs_total_edges_dict)

    print("===================== num of edge per region=================")
    print(sum_of_edges_per_region)

    region_2_region_table_EXPECTED_df = calc_expected_values_for_chi_2(region_2_region_table_edges_not_normalized)
    region_OUT_vs_total_edges_EXPECTED_dict = calculate_OUT_vs_total_edges("", region_2_region_table_EXPECTED_df)
    check_chi_test(region_OUT_vs_total_edges_dict, region_OUT_vs_total_edges_EXPECTED_dict)

    distinct_author_edges_per_region = os.path.join(paths["results_folder"], "distinct_author_edges_per_region.xlsx")
    distinct_author_edges_per_region_DETAILED = os.path.join(paths["results_folder"], "distinct_author_edges_per_region_DETAILED.xlsx")
    _, region_to_authors_detailed, regions_authors_2_distinct_authors_dict = build_distinct_author_edges_per_region(
        author_2_author_table,
        paths["Biography_path"],
        paths["authors_words_number_path"],
        distinct_author_edges_per_region,
        distinct_author_edges_per_region_DETAILED,
        False
    )
    region_distinct_authors_RATIO_dict = plot_distinct_author_edges(region_to_authors_detailed)
    distinct_authors_num = convert_distinct_author_dict_2_total(region_to_authors_detailed)

    plt.rcParams.update({'font.size': 15})
    region_distinct_authors_ratio_df = pd.DataFrame.from_dict(distinct_authors_num, orient='index')
    plot_region_2_region_table("", "BuGn", "", True, region_distinct_authors_ratio_df)
    plt.rcParams.update({'font.size': 10})

    region_distinct_authors_EXPECTED_df = calc_expected_values_for_chi_2("", region_distinct_authors_ratio_df)
    region_distinct_authors_EXPECTED_dict = {
        index: {col: row[col] for col in region_distinct_authors_EXPECTED_df.columns}
        for index, row in region_distinct_authors_EXPECTED_df.iterrows()
    }

    region_distinct_authors_RATIO_EXPECTED_dict = calculate_distinct_author_edges_RATIO_EXPECTED(region_distinct_authors_EXPECTED_dict)
    check_chi_test(region_distinct_authors_RATIO_dict, region_distinct_authors_RATIO_EXPECTED_dict)

    plot_dicts(region_OUT_vs_total_edges_dict, region_distinct_authors_RATIO_dict, 'Citations’ external out-degree', 'Resource diversity', "")

    print(regions_authors_2_distinct_authors_dict)
    print(regions_authors_num_of_references_dict)

    regions_authors_2_distinct_authors_dict = convert_2_ratio(regions_authors_2_distinct_authors_dict)
    regions_authors_num_of_references_dict = convert_2_ratio(regions_authors_num_of_references_dict)

    num_of_bars = -1
    plot_region_bars_single_figure(regions_authors_2_distinct_authors_dict, num_of_bars, False)
    plot_region_bars_single_figure(regions_authors_num_of_references_dict, num_of_bars, False)

    plot_authors_ration_X_Y(regions_authors_num_of_references_dict, regions_authors_2_distinct_authors_dict)

    network_regions_satistics_df, _ = calc_network_regions_satistics(
        paths["Biography_path"],
        paths["authors_words_number_path"],
        paths["DB_final_edges_EN"],
        paths["DB_nodes_path"]
    )
    sum_of_edges_per_region.name = "Number of edges per region"
    sum_of_edges_per_region = sum_of_edges_per_region.to_frame()
    sum_of_edges_per_region.index.name = 'REGION'
    network_regions_satistics_df = pd.merge(network_regions_satistics_df, sum_of_edges_per_region, on="REGION")

    network_regions_satistics_df['Number of edges per region'] = pd.to_numeric(network_regions_satistics_df['Number of edges per region'], errors='coerce')

    sum_row = network_regions_satistics_df.sum(numeric_only=True)
    network_regions_satistics_df = pd.concat([network_regions_satistics_df, pd.DataFrame([sum_row], index=['Total'])])

    region_2_region_table_normalized_by_word_count_df = region_2_region_table_normalized_by_word_count_df['Text citation rate']
    region_2_region_table_normalized_by_word_count_df *= 100
    print(region_2_region_table_normalized_by_word_count_df)

    network_regions_satistics_df.index.name = 'REGION'
    network_regions_satistics_df = pd.merge(network_regions_satistics_df, region_2_region_table_normalized_by_word_count_df, on="REGION")
    print(network_regions_satistics_df)

    entire_network_author_citation_rate = network_regions_satistics_df.loc["Total", "Number of edges per region"] / network_regions_satistics_df.loc["Total", "Number of authors per region"]
    region_2_region_full_total_df.loc["Total"] = entire_network_author_citation_rate
    network_regions_satistics_df = pd.merge(network_regions_satistics_df, region_2_region_full_total_df, on="REGION")
    print(network_regions_satistics_df)

    region_out_citation_df = pd.read_excel(region_2_region_table_edges_not_normalized, engine='openpyxl', index_col=0)
    region_out_citation_df = region_out_citation_df[['total']]
    region_out_citation_df.index.name = 'REGION'
    network_regions_satistics_df = pd.merge(network_regions_satistics_df, region_out_citation_df, on="REGION")
    network_regions_satistics_df.rename(columns={'total': 'total_region_out_citation'}, inplace=True)
    print(network_regions_satistics_df)


if __name__ == "__main__":
    main()