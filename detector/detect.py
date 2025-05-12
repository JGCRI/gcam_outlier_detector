import pandas as pd
import matplotlib.pyplot as plt
import gcamreader
import os
import re
import logging
import concurrent.futures
import random
import lxml.etree as ET
from collections import defaultdict

logger = logging.getLogger(__name__)

class Detector:

    def __init__(self, db_path, queries, x_label, divisor_label=None, mapping_file=None, x_func=None,
                 scenario_index=0, aggregate='max', threshold=2, interval=1, historical_year_end=2020, 
                 include_normal=False, maxMemory="8g"):
        """Initialize the Detector class.
        
        Parameters
        ----------
        db_path : str
            Path to the GCAM database.
        queries : str
            Path to the queries file with all needed queries.
        x_label : str
            The query label for the x-axis variable or the variable to detect against.
        divisor_label : str, optional
            The query label for the divisor, everything is divided by this value, by default None
        mapping_file : str, optional
            Path to the mapping file, by default None.
        x_func : function, optional
            Function to calculate the values in the x axis, a default function is provided.
        scenario_index : int, optional
            The index of the scenario to use, by default 0 or the first scenario in the database.
        aggregate : str, optional
            The aggregate function to use during detection, by default 'max' of historical value. 
        threshold : int, optional
            The threshold for the anomaly detection, by default 2.
        interval : int, optional
            The interval for the anomaly detection, by default 1.
        historical_year_end : int, optional
            The end year for the historical data, by default 2020.
        include_normal : bool, optional
            Whether to include normal data in the output, by default False.
        maxMemory : str, optional
            The maximum memory to use for the database connection, by default "8g".
        """
        self.connection = gcamreader.LocalDBConn(os.path.dirname(db_path), os.path.basename(db_path), maxMemory=maxMemory)
        self.scenarios = self.connection.listScenariosInDB()
        self.include_normal = include_normal
        self.divisor_label = divisor_label
        if not isinstance(historical_year_end, int):
            raise ValueError("historical_year_end must be an integer.")
        self.historical_year_end = historical_year_end
        if mapping_file is None:
            self.mapping_df = None
        if x_func is None:
            x_func = self._default_calculate_x
        if not self.update_data(queries=queries, x_label=x_label, scenario_index=scenario_index, threshold=threshold,
                                interval=interval, mapping_file=mapping_file, x_func=x_func, aggregate=aggregate):
            raise ValueError("Failed to initialize data.")
        

    def _get_query_index(self, index):
        """Get the query at the specified index.
        
        Parameters
        ----------
        index : int
            The index of the query to retrieve.
        
        Returns
        -------
        Query
            The query object at the specified index.
        """
        if index >= len(self.queries) or index < 0:
            logger.error("Invalid query number passed.")
            return None
        query = self.queries[index]
        return query
    
    def _get_query_label(self, label):
        """Get the query with the specified label.

        Parameters
        ----------
        label : str
            The label of the query to retrieve.

        Returns
        -------
        Query
            The query object with the specified label.
        """
        query = next((q for q in self.queries if q.title == label), None)
        if query is None:
            logger.error(f"Query with label {label} not found.")
        return query
    
    def _get_df(self, query, scenario_index=None):
        """Get the DataFrame for the specified query and scenario index.

        Parameters
        ----------
        query : Query
            The query object to retrieve data for.
        scenario_index : int, optional
            The index of the scenario to use, by default None. If not provided, 
            the current scenario index is used (or one given to class).

        Returns
        -------
        DataFrame
            The DataFrame containing the data for the specified query and scenario index.
        """
        if scenario_index is None:
            scenario_index = self.scenario_index
        if scenario_index >= len(self.scenarios['fqName']) or scenario_index < 0:
            logger.error("Invalid scenario number passed.")
            return None
        scenario = [self.scenarios['fqName'][scenario_index]]
        df = self.connection.runQuery(query, scenarios=scenario)
        if df is None:
            logging.warning(f"Query {query.title} returned no data.")
            return None
        return df
    
    def _get_query_df(self, label):
        """Get the DataFrame for the specified query label.

        Parameters
        ----------
        label : str
            The label of the query to retrieve data for.
        
        Returns
        -------
        DataFrame
            The DataFrame containing the data for the specified query label.
        """
        query = self._get_query_label(label)
        if query is None:
            return None
        df = self._get_df(query)
        if df is None:
            return None
        df = df.rename(columns={'value': label})
        return df
    
    def _default_calculate_x(self):
        """Calculate the x-axis values based on the specified x_label and divisor_label.
        If divisor_label is provided, the x-axis values are calculated as 
        (x_label * 1000000) / (divisor_label * 1000) else they are calculated as (x_label * 1000000). 

        Returns
        -------
        DataFrame
            The DataFrame containing the x-axis values with the corresponding region and year.
        """
        y_df = self._get_query_df(self.x_label)
        if y_df is None:
            return y_df
        if self.divisor_label:
            divisor_df = self._get_query_df(self.divisor_label)
            if divisor_df is None:
                return None
            final_df = pd.merge(y_df, divisor_df, on=['region', 'Year'], how='inner',
                                   suffixes=(self.x_label, self.divisor_label))
            final_df['x_var'] = (final_df[self.x_label] * 1000000) / (final_df[self.divisor_label] * 1000)
        else:
            final_df = y_df
            final_df['x_var'] = (final_df[self.x_label] * 1000000)
        return final_df

    def _aggregate_values(self, value, mapping_df):
        """Aggregate the values based on the mapping DataFrame.

        Parameters
        ----------
        value : str
            The value to aggregate.
        mapping_df : DataFrame
            The mapping DataFrame containing the sector and category mappings.
        
        Returns
        -------
        str
            The aggregated category for the value.
        """
        for _, row in mapping_df.iterrows():
            pattern = row['sector']
            if re.search(pattern, value):
                return row['category']
        return 'other'
        
    def _check_file(self, file):
        """Check if the specified file exists.
        
        Parameters
        ----------
        file : str
            The path to the file to check.

        Returns
        -------
        bool
            True if the file exists, False otherwise.    
        """
        if not os.path.exists(file):
            logger.error(f"File {file} does not exist.")
            return False
        return True

    def update_data(self, queries=None, x_label=None, divisor_label=None, scenario_index=None,
                    mapping_file=None, x_func=None, threshold=None, interval=None, aggregate=None):
        """Update the data for the Detector class.

        Parameters
        ----------
        queries : str, optional
            Path to the queries file with all needed queries, by default None.
        x_label : str, optional
            The query label for the x-axis variable or the variable to detect against, by default None.
        divisor_label : str, optional
            The query label for the divisor, everything is divided by this value, by default None.
        scenario_index : int, optional
            The index of the scenario to use, by default None.
        mapping_file : str, optional
            Path to the mapping file, by default None.
        x_func : function, optional
            Function to calculate the values in the x axis by default None.
        threshold : int, optional
            The threshold for the anomaly detection, by default None.
        interval : int, optional
            The interval for the anomaly detection, by default None.
        aggregate : str, optional
            The aggregate function to use during detection, by default None.
        
        Returns
        -------
        bool
            True if the data was updated successfully, False otherwise.
        """
        success = False
        if queries:
            if not self._check_file(queries):
                return success
            self.queries = gcamreader.parse_batch_query(queries)
        if mapping_file:
            if not self._check_file(mapping_file):
                return success
            self.mapping_df = pd.read_csv(mapping_file)
        if scenario_index is not None:
            if scenario_index >= len(self.scenarios['fqName']) or scenario_index < 0:
                logger.error("Invalid scenario index passed.")
                return success
            self.scenario_index = scenario_index
        if x_label is not None:
            self.x_label = x_label
        if divisor_label is not None:
            self.divisor_label = divisor_label
        if threshold is not None:
            self.threshold = threshold
        if interval is not None:
            self.interval = interval
        if x_func is not None:
            if not callable(x_func):
                logger.error("X function is not callable.")
                return success
            self.x_func = x_func
        if aggregate is not None:
            if aggregate not in ['max', 'min', 'mean', 'median', 'low-quart', 'high-quart']:
                logger.error(f"Invalid aggregate function passed: {aggregate}.")
                return success
            self.aggregate = aggregate
        self.data = self.x_func()
        if self.data is None:
            logger.error("Failed to calculate x value.")
            return success
        success = True
        return success

    def check_scenarios(self):
        """Check the number of scenarios in the database.
        
        Returns
        -------
        int
            The number of scenarios in the database.
        """
        num_scenarios = len(self.scenarios)
        return num_scenarios
    
    def calc_aggregate(self, df):
        """Calculate the aggregate value for the specified DataFrame.
        
        Parameters
        ----------
        df : DataFrame
            The DataFrame with y_var to calculate the aggregate value for.

        Returns
        -------
        float
            The calculated aggregate value.
        """
        ret = None
        if self.aggregate == 'max':
            ret = df['y_var'].max()
        elif self.aggregate == 'min':
            ret = df['y_var'].min()
        elif self.aggregate == 'mean':
            ret = df['y_var'].mean()
        elif self.aggregate == 'median':
            ret = df['y_var'].median()
        elif self.aggregate == 'low-quart':
            ret = df['y_var'].quantile(0.25)
        elif self.aggregate == 'high-quart':
            ret = df['y_var'].quantile(0.75)
        else:
            logger.error(f"Invalid aggregate function passed: {self.aggregate}.")
        return ret

    def list_queries(self, queries):
        """List the queries in the specified file.
        
        Parameters
        ----------
        queries : str
            Path to the queries file with all needed queries.
        """
        query_lst = gcamreader.parse_batch_query(queries)
        for i in range(len(query_lst)):
            print(f"{i}: {query_lst[i].querystr}")
    
    def compare(self, stat, cmp, low, high, filter=None):
        """Compare the specified statistic with the given bounds.
        
        Parameters
        ----------
        stat : float
            The statistic to compare.
        cmp : DataFrame
            The DataFrame with the y_var to compare against.
        low : float
            The lower bound for the comparison. If None, no lower bound is used.
        high : float
            The upper bound for the comparison. If None, no upper bound is used.
        filter : Series, optional
            A boolean mask to filter the DataFrame, by default None.

        Returns
        -------
        Series
            A boolean mask indicating which rows in the DataFrame meet the comparison criteria.
        """
        mask = None
        if stat is not None:
            if low or high:
                if low is None:
                    mask = (stat * high) >= cmp['y_var']
                elif high is None:
                    mask = (stat * low) <= cmp['y_var']
                else:
                    mask = (((stat * high) >= cmp['y_var']) &
                            ((stat * low) <= cmp['y_var']))    
            if filter is not None:
                mask = mask & filter
        return mask
    
    def get_masks(self, stat, df, filter=None):
        """
        Get the masks for the specified statistic and DataFrame. Three 
        masks are returned depending on the threshold and interval values.

        Parameters
        ----------
        stat : float
            The statistic to compare.
        df : DataFrame
            The DataFrame with the y_var to compare against.
        filter : Series, optional
            A boolean mask to filter the DataFrame, by default None.
        
        Returns
        -------
        tuple
            A tuple containing three boolean masks for low, medium, and high 
            anomalies in the df.
        """
        lbound = self.threshold
        hbound = self.threshold + self.interval
        mask_low = self.compare(stat, df, lbound, hbound, filter)
        if mask_low is None:
            return None, None, None
        lbound = hbound
        hbound = lbound + self.interval
        mask_med = self.compare(stat, df, lbound, hbound, filter)
        lbound = hbound
        hbound = None
        mask_high = self.compare(stat, df, lbound, hbound, filter)
        return mask_low, mask_med, mask_high

    def print_report(self, report):
        """
        Print the anomaly report.
        
        Parameters
        ----------
        report : dict
            The anomaly report containing the detected anomalies.
        """
        def print_level(level, values):
            anomaly_text = f"{level.capitalize()} Anomalies"
            print(anomaly_text)
            print("-" * len(anomaly_text))
            print()
            for value in values:
                print(f"{value[0]}, {value[1]}, {value[2]}")
            print()
        print()
        print("Anamoly Report:")
        print()
        print("Format: Query Title, Category, Income Level")
        print()
        for level, values in report.items():
            print_level(level, values)
    
    def get_mapped_query_df(self, query, scenario_index=None):
        """
        Get the mapped DataFrame for the specified query and scenario index.

        Parameters
        ----------
        query : Query
            The query object to retrieve data for.
        scenario_index : int, optional
            The index of the scenario to use, by default None.
        
        Returns
        -------
        DataFrame
            The mapped DataFrame containing the data for the specified query and scenario index.
        """
        query_df = self._get_df(query=query, scenario_index=scenario_index)
        if query_df is None:
            return None
        query_label = query.title
        query_df = query_df.rename(columns={'value': query_label})
        root = ET.fromstring(query.querystr)
        mapping_var = root.find('mappingVariable')
        if mapping_var:
            mapping_var = mapping_var.get('name')
        elif self.mapping_df is not None:
            if self.mapping_df.columns[0] in query_df.columns:
                mapping_var = self.mapping_df.columns[0]
                query_df['category'] = query_df[mapping_var].apply(self._aggregate_values, args=(self.mapping_df,))
        if not mapping_var:
            mapping_var = root.find('axis1').get('name')
        if 'category' not in query_df:
            query_df = query_df.rename(columns={mapping_var: 'category'})
        query_df = query_df.groupby(['Year', 'region', 'category'])[query_label].sum().reset_index()
        return query_df

    def detect(self, query, scenario_index=None, graph=False):
        """
        Detect anomalies in the specified query and scenario index. This function 
        contains the primary logic for detecting anomalies in the data. 

        Parameters
        ----------
        query : Query
            The query object to detect anomalies for.
        scenario_index : int, optional
            The index of the scenario to use, by default None.
        graph : bool, optional
            Whether to save a graph of the detected anomalies, by default False.

        Returns
        -------
        dict
            A dictionary containing the detected anomalies, with keys for low, medium, 
            and high anomalies.
        """
        report = None
        if query is None or query.title == self.x_label or query.title == self.divisor_label:
            return report
        query_df = self.get_mapped_query_df(query, scenario_index)
        if query_df is None:
            raise ValueError("Failed to get query data.")
        query_label = query.title
        data = pd.merge(query_df, self.data, how='inner', on=['Year', 'region'])
        data['y_var'] = (data[query_label] * 1000000)
        if self.divisor_label:
            data['y_var'] /= (data[self.divisor_label] * 1000)
        quantiles = data['x_var'].quantile([0.3333, 0.6667])
        bins = [data['x_var'].min(), quantiles[0.3333], quantiles[0.6667], data['x_var'].max()]
        labels = ['Low Income', 'Mid Income', 'High Income']
        data['bin'] = pd.cut(data['x_var'], bins=bins, labels=labels, include_lowest=True)
        data['anamoly'] = 'Historical'
        
        report = defaultdict(list)

        category_dfs = data.groupby('category')
        category_dfs = {category: group for category, group in category_dfs}
        
        for category, df in category_dfs.items():
            label_dfs = df.groupby('bin', observed=False)
            label_dfs = {label: group for label, group in label_dfs}
            gen_dfs = []
            if graph:
                plt.figure(figsize=(20, 14))
            for label, label_df in label_dfs.items():
                hist_mask = label_df['Year'] <= 2020
                hist_df = label_df[hist_mask]
                gen_df = label_df[~hist_mask]
                stat = self.calc_aggregate(hist_df)
                mask_low, mask_med, mask_high = self.get_masks(stat, gen_df)
                if mask_low is None:
                    continue
                regions = gen_df['region'].unique().tolist()
                outlier_regions = {}
                for region in regions:
                    mask_region = gen_df['region'] == region
                    rem_df = gen_df[~mask_region]
                    stat = self.calc_aggregate(rem_df)
                    temp_mask_low, temp_mask_med, temp_mask_high = self.get_masks(stat, gen_df, mask_region)
                    if mask_low is None:
                        continue
                    mask_low |= temp_mask_low
                    mask_med |= temp_mask_med
                    mask_high |= temp_mask_high
                    temp_total = temp_mask_low | temp_mask_med | temp_mask_high
                    if any(temp_total):
                        outlier_regions[region] = random.choice([i for i, val in enumerate(temp_total) if val])
                mask_low &= ~(mask_med | mask_high)
                mask_med &= ~mask_high

                mask_total = mask_low | mask_med | mask_high
                detected = True if any(mask_total) else False
                mask_not = ~mask_total 
        
                non_anm_gen_df = gen_df[mask_not]
                low_anm_gen_df = gen_df[mask_low]
                med_anm_gen_df = gen_df[mask_med]
                high_anm_gen_df = gen_df[mask_high]

                
                gen_df.loc[mask_not, 'anamoly'] = 'No Anamoly'
                gen_df.loc[mask_low, 'anamoly'] = 'Low Anamoly'
                gen_df.loc[mask_med, 'anamoly'] = 'Med Anamoly'
                gen_df.loc[mask_high, 'anamoly'] = 'High Anamoly'
        
                gen_dfs.append(gen_df)

                if detected:
                    if mask_high.any():
                        report['high'].append([query_label, category, label])
                    elif mask_med.any():
                        report['med'].append([query_label, category, label])
                    else:
                        report['low'].append([query_label, category, label])
                else:
                    report['no'].append([query_label, category, label])
                
                if graph:
                    plt.scatter(hist_df['x_var'], hist_df['y_var'], color='blue', alpha=0.5)
                    plt.scatter(non_anm_gen_df['x_var'], non_anm_gen_df['y_var'], color='#03ff7c', alpha=0.5)
                    plt.scatter(low_anm_gen_df['x_var'], low_anm_gen_df['y_var'], color='#a0b310', alpha=0.5)
                    plt.scatter(med_anm_gen_df['x_var'], med_anm_gen_df['y_var'], color='#8c680d', alpha=0.5)
                    plt.scatter(high_anm_gen_df['x_var'], high_anm_gen_df['y_var'], color='#8c1a0d', alpha=0.5)
                    for region, index in outlier_regions.items():
                        plt.annotate(region, (gen_df.iloc[index]['x_var'], gen_df.iloc[index]['y_var']), 
                                     textcoords="offset points", xytext=(10,10), ha='center')
            
            combined_df = pd.concat(gen_dfs)
            if self.include_normal:
                combined_df = pd.concat([combined_df, hist_df])
            combined_df = combined_df.sort_values(by=['region', 'Year']).reset_index(drop=True)

            if graph:
                plt.axvline(x=bins[0], color='red', linestyle='--', label='income bounds', linewidth=1.5, alpha=0.6)
                for bin in bins[1:]:
                    plt.axvline(x=bin, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
                
                plt.scatter([], [], color='blue', alpha=0.5, label='historical')
                plt.scatter([], [], color='#03ff7c', alpha=0.5, label='generated -- non anamalous')
                plt.scatter([], [], color='#a0b310', alpha=0.5, label='generated -- low anomaly')
                plt.scatter([], [], color='#8c680d', alpha=0.5, label='generated -- moderate anomaly')
                plt.scatter([], [], color='#8c1a0d', alpha=0.5, label='generated -- high anomraly')
                #plt.title(f"NOX Emissions data for {sectors=}")
                x_label = f"{self.x_label}"
                y_label = f"{query_label}"
                if self.divisor_label:
                    x_label += f"/{self.divisor_label}"
                    y_label += f"/{self.divisor_label}"
                
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                
                plt.legend()
                
                plt.grid(True)
                graph_name = f'{category}_{self.x_label}_{self.divisor_label}_{query_label}.jpg'
                plt.savefig(f'graphs/{graph_name}')
                print(f"graph saved: {graph_name}")
        
            csv_name = f'{category}_{query_label}_output.csv'
            combined_df.rename(columns={'x_var': 'GDP/Capita', 'y_var': 'Emissions/Capita'})
            if not self.include_normal:
                combined_df = combined_df[combined_df['anamoly'] != 'No Anamoly']
            combined_df.to_csv(f'csvs/{csv_name}', index=False)
            print(f"csv saved: {csv_name}")
        

        print()
        return None if not report else report
        
    def detect_all(self, max_workers=None, graph=False):
        """
        Detect anomalies in all queries using parallel processing.

        Parameters
        ----------
        max_workers : int, optional
            The maximum number of workers to use for parallel processing, by default None.
        graph : bool, optional
            Whether to save a graph of the detected anomalies, by default False.

        Returns
        -------
        dict
            A dictionary containing the detected anomalies for all queries, with keys for low, medium, 
            and high anomalies.
        """
        report = None
        if max_workers is None:
            logger.error("Max workers not set.")
            return report
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        futures = [executor.submit(self.detect, query=q, graph=graph) for q in self.queries]
        report = defaultdict(list)
        for future in concurrent.futures.as_completed(futures):
            ret = future.result()
            if ret:
                for key, value in ret.items():
                    report[key].extend(value)
        executor.shutdown()
        return None if not report else report
    
# change labels in csvs

# n-1 region check nox other so check the distributions within each bin without a given region
# for all regions so that you can maybe tell if a region itself is anomalous 

# run on BYU database

# cluster the histories together and look for anamolies if generated data doesnt fit clusters