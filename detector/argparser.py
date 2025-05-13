import argparse
import os

parser = argparse.ArgumentParser(description="GCAM Anamoly detector!")

parser.add_argument(
    "-d",
    "--db-path",
    type=str,
    help="Path to the database file",
)

parser.add_argument(
    "-q",
    "--queries",
    type=str,
    help="Path to the queries file. Default is 'queries.xml' in the current directory",
    default="queries.xml",
)

parser.add_argument(
    "-x",
    "--x-label",
    type=str,
    help="The label for the query plotted on the x-axis. Default is 'GDP'.",
    default="GDP",
)

parser.add_argument(
    "-l",
    "--divisor-label",
    type=str,
    help="The label for the query which will divide all values plotted. Default is 'Population' for per capita.",
    default="Population",
)

parser.add_argument(
    "-n",
    "--query-index",
    type=int,
    help="Query index to run from the queries file. Default is all queries.",
    default=-1,
)

parser.add_argument(
    "-u",
    "--query-label",
    type=str,
    help="Query label to run from the queries file. Default is all queries.",
    default="",
)

parser.add_argument(
    "-m",
    "--mapping-file",
    type=str,
    help="Path to the mapping file. Default is 'mapping_file.csv' in the current directory",
    default="mapping_file.csv",
)

parser.add_argument(
    "-s",
    "--scenario-index",
    type=int,
    help="Scenario index to run. Default is the first scenario or 0.",
    default=0,
)

parser.add_argument(
    "-t",
    "--threshold",
    type=float,
    help="Threshold to use for the detector. Threshold refers to how much larger the anamoly has to be compared to "
    "historical data for it to be flagged. Default is 2 or 2 times the aggregate of historical data.",
    default=2,
)

parser.add_argument(
    "-i",
    "--interval",
    type=float,
    help="Interval to use for the detection ranges after threshold. Default is 1 so anamolies are detected in the "
    "range of 2 to 3, 3 to 4, etc.",
    default=1,
)

parser.add_argument(
    "-a",
    "--aggregate",
    type=str,
    help="Aggregate function to use. Default is 'max'. Other options are 'min', 'mean', 'median'," 
    "'low-quart', 'high-quart'.",
    default="max",
)

parser.add_argument(
    "-g",
    "--graph",
    action="store_true",
    help="Whether to save the graphs for the results or not. Default is False.",
    default=False,
)

cpu_count = os.cpu_count()

parser.add_argument(
    "-w",
    "--workers",
    type=int,
    help="Number of workers to use for the detector when detecting across multiple queries. "
    "Default is local cpu count - 2.",
    default= 1 if cpu_count - 2 < 1 else cpu_count - 2,
)

parser.add_argument(
    "-y",
    "--historical-year-end",
    type=int,
    help="The end year of the historical data. Default is 2020.",
    default=2020,
)

parser.add_argument(
    "--graph-path",
    type=str,
    help="Path to save the graphs. Default is the current directory/graphs.",
    default="graphs",
)

parser.add_argument(
    "--csv-path",
    type=str,
    help="Path to save the csv files with the results. Default is the current directory/csvs.",
    default="csvs",
)

parser.add_argument(
    "--include-normal",
    action="store_true",
    help="Whether to include normal data in the outputs or not. Default is False.",
    default=False,
)

