"""
Enhanced CSV Data Analytics MCP Server

This server provides tools for reading, analyzing, and performing data quality checks on CSV files via FastMCP.

Available tools:
- read_csv: Preview CSV contents and structure.
- aggregate_file: Group and aggregate data by columns.
- check_missing_values: Identify columns with missing data.
- check_duplicates: Analyze duplicate rows.
- check_data_types: Detect mixed or inconsistent column types.
- check_date_formats: Find inconsistent date formats.
- check_categorical_frequency: Analyze rare or dominant categorical values.
- check_numeric_format: Detect numeric formatting issues.
- check_date_outliers: Find outlier dates (very old, far future, etc).
- check_category_standardization: Standardize categorical values using a mapping.
- check_primary_keys: Find columns that could serve as primary keys.
- propose_table_name: Suggest a business-friendly table name.
- check_outliers_numbers: Detect numeric outliers using IQR.
- remove_duplicates: Remove duplicate rows and save cleaned file.
- fill_missing_values_zero: Fill missing values with zero and save cleaned file.
- standardize_column_names: Standardize column names to snake_case.
- standardize_numeric_formats: Standardize numeric formats (commas, periods).
- standardize_date_formats: Standardize date columns to yyyy-mm-dd.

Example usage:
- /read_csv file_path="sample.csv"
- /aggregate_file file_path="sample.csv" group_by="Category" agg_column="Sales_Amount" agg_function="sum"
- /aggregate_file file_path="sample.csv" group_by="Region,Customer_Type" agg_column="Units_Sold" agg_function="mean"
- /check_missing_values file_path="sample.csv"
- /check_duplicates file_path="sample.csv"
- /check_data_types file_path="sample.csv"
- /check_date_formats file_path="sample.csv"
- /check_categorical_frequency file_path="sample.csv"
- /check_numeric_format file_path="sample.csv"
- /check_date_outliers file_path="sample.csv"
- /check_category_standardization file_path="sample.csv" category_mapping={"Heinken": "Heineken", "HNK": "Heineken"}
- /check_primary_keys file_path="sample.csv"
- /propose_table_name file_path="sample.csv"
- /check_outliers_numbers file_path="sample.csv"
- /remove_duplicates file_path="sample.csv"
- /fill_missing_values_zero file_path="sample.csv"
- /standardize_column_names file_path="sample.csv"
- /standardize_numeric_formats file_path="sample.csv"
- /standardize_date_formats file_path="sample.csv"
"""

from fastmcp import FastMCP
import pandas as pd
from pathlib import Path
import re
from datetime import datetime, timedelta
import warnings
from dateutil.parser import parse

mcp = FastMCP("from-chaos-to-cube-server")

@mcp.tool()
def read_csv(file_path: str) -> str:
    """
    Reads a CSV file and returns its contents and basic info.
    
    Use when: analyzing data files, checking CSV structure, or viewing data samples.
    Examples: 'read sales.csv', 'analyze the data file', 'show me what's in the CSV'
    
    Args:
        file_path: Path to the CSV file to read. Can be absolute or relative.
    
    Returns:
        String containing file info and preview of the data.
    """
    try:
        file_obj = Path(file_path)
        
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        
        result = f"Successfully read CSV: {file_obj}\n"
        result += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        result += "First 10 rows:\n"
        result += df.head(10).to_string()
        
        return result
        
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

@mcp.tool()
def aggregate_file(file_path: str, group_by: str, agg_column: str, agg_function: str) -> str:
    """
    Aggregates data by grouping columns and applying aggregation functions.
    
    Use when: calculating totals, averages, counts, or other statistics by category.
    Examples: 'sum sales by country', 'average units sold by category', 'count products by type'.
    
    Args:
        file_path: Path to the CSV file to aggregate
        group_by: Column names to group by. Use comma-separated for multiple columns (e.g., 'Category,Region')
        agg_column: Column name to aggregate
        agg_function: Aggregation function to apply: sum, mean, count, min, max, std
    
    Returns:
        String containing aggregation results and summary statistics.
    """
    try:
        file_obj = Path(file_path)
        agg_function = agg_function.lower()
        
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        group_columns = [col.strip() for col in group_by.split(',')]
    
        valid_functions = ['sum', 'mean', 'count', 'min', 'max', 'std']
        if agg_function not in valid_functions:
            return f"Error: Invalid function '{agg_function}'. Valid options: {valid_functions}"
        
        if agg_function == 'count':
            agg_result = df.groupby(group_columns).size().reset_index(name='count')
            agg_col_name = 'count'
        else:
            agg_result = df.groupby(group_columns)[agg_column].agg(agg_function).reset_index()
            agg_col_name = agg_column
        
        result = f"Aggregation Results:\n"
        result += f"File: {file_obj}\n"
        result += f"Grouped by: {', '.join(group_columns)}\n"
        result += f"Aggregation: {agg_function}({agg_column if agg_function != 'count' else 'rows'})\n\n"
        
        result += agg_result.to_string(index=False)
        
        if agg_function != 'count':
            total = agg_result[agg_col_name].sum() if agg_function in ['sum', 'mean'] else None
            if total is not None:
                result += f"\n\nTotal {agg_function}: {total:,.2f}"
        
        return result
        
    except Exception as e:
        return f"Error aggregating CSV: {str(e)}"

@mcp.tool()
def check_missing_values(file_path: str) -> str:
    """
    Analyze CSV file to identify all columns with missing values and calculate statistics.
    
    Use when: checking data quality, identifying incomplete records, data cleaning preparation.
    Examples: 'check for missing data', 'find null values', 'analyze data completeness'
    
    Args:
        file_path: Path to the CSV file to analyze for missing values
    
    Returns:
        String containing detailed missing value analysis including counts and percentages.
    """
    try:
        file_obj = Path(file_path)
        
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        missing_counts = df.isnull().sum()
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        
        columns_with_missing = missing_counts[missing_counts > 0]
        
        result = f"Missing Values Analysis:\n"
        result += f"File: {file_obj}\n"
        result += f"Total rows: {len(df):,}\n"
        result += f"Total columns: {len(df.columns)}\n\n"
        
        if len(columns_with_missing) == 0:
            result += "No missing values found in any column!\n"
        else:
            result += f"Found missing values in {len(columns_with_missing)} column(s):\n\n"
            
            result += f"{'Column':<20} {'Missing Count':<15} {'Percentage':<12}\n"
            result += "-" * 50 + "\n"
            
            for column in columns_with_missing.index:
                count = missing_counts[column]
                percentage = missing_percentages[column]
                result += f"{column:<20} {count:<15,} {percentage:<12.2f}%\n"
            
            total_missing = columns_with_missing.sum()
            result += "\n" + "="*50 + "\n"
            result += f"Total missing values: {total_missing:,}\n"
            result += f"Columns affected: {len(columns_with_missing)}/{len(df.columns)}\n"
            result += f"Most affected column: {columns_with_missing.idxmax()} ({columns_with_missing.max():,} missing)\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing missing values: {str(e)}"

@mcp.tool()
def check_duplicates(file_path: str, subset: list[str] = None, ignore_columns: list[str] = None, preview_limit: int = 5) -> str:
    """
    Checks a CSV file for duplicate rows and provides detailed duplicate analysis.
    
    Use when: validating data uniqueness, identifying data entry errors, cleaning datasets.
    Examples: 'check for duplicate rows', 'find repeated records', 'analyze data duplicates'
    
    Args:
        file_path: Path to the CSV file to check for duplicates
        subset: List of columns to consider for duplicate detection (default: all columns)
        ignore_columns: Columns to exclude from duplicate detection
        preview_limit: Number of duplicate samples to display
    
    Returns:
        String containing duplicate analysis including total count, affected rows, and preview.
    """
    
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        # Load CSV
        df = pd.read_csv(file_obj)
        total_rows = len(df)
        
        # Handle ignore columns
        if ignore_columns:
            df = df.drop(columns=[col for col in ignore_columns if col in df.columns], errors="ignore")
        
        # Detect duplicates
        duplicated_mask = df.duplicated(subset=subset, keep=False)
        duplicates_df = df[duplicated_mask]
        duplicate_count = duplicates_df.shape[0]
        
        result = f"Duplicate Rows Analysis\n"
        result += f"File: {file_obj}\n"
        result += f"Total rows: {total_rows:,}\n\n"
        
        if duplicate_count == 0:
            result += "No duplicate rows found!\n"
            return result
        
        # Count unique duplicate sets
        duplicate_groups = duplicates_df.groupby(list(df.columns)).size().reset_index(name="count")
        duplicate_groups = duplicate_groups[duplicate_groups["count"] > 1]
        
        duplicate_percentage = (duplicate_count / total_rows) * 100
        unique_rows = total_rows - duplicate_count
        
        result += f"Found {duplicate_count:,} duplicate row(s)\n"
        result += f"Duplicate percentage: {duplicate_percentage:.2f}%\n"
        result += f"Unique rows: {unique_rows:,}\n\n"
        
        # Show duplicate groups
        result += f"Duplicate groups (rows appearing more than once): {len(duplicate_groups)}\n"
        
        if len(duplicate_groups) <= preview_limit:
            result += duplicate_groups.to_string(index=False)
        else:
            result += f"Showing first {preview_limit} groups:\n"
            result += duplicate_groups.head(preview_limit).to_string(index=False)
        
        return result
    
    except Exception as e:
        return f"Error detecting duplicates: {str(e)}"

    

@mcp.tool()
def check_data_types(file_path: str) -> str:
    """
    Checks columns for mixed or incorrect data types and suggests potential corrections.
    
    Use when: validating consistency of column types before analysis or cleaning.
    Examples: 'check column types', 'detect data type issues', 'verify numeric and string columns'
    
    Args:
        file_path: Path to the CSV file to check.
    
    Returns:
        String summary of columns with inconsistent or unexpected data types.
    """
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        result = f"Data Type Analysis for file: {file_obj}\n\n"
        issues_found = False
        
        for col in df.columns:
            unique_types = df[col].map(type).unique()
            if len(unique_types) > 1:
                issues_found = True
                types_str = ", ".join([t.__name__ for t in unique_types])
                result += f"Column '{col}' has mixed types: {types_str}\n"
        
        if not issues_found:
            result += "All columns have consistent data types.\n"
        
        return result
    
    except Exception as e:
        return f"Error checking data types: {str(e)}"

@mcp.tool()
def check_date_formats(file_path: str) -> str:
    """
    Detects columns that are datetime-like and checks for inconsistent date formats.
    
    Use when: validating date/time columns before normalization.
    Examples: 'check date formats', 'identify inconsistent dates', 'analyze datetime columns'
    
    Args:
        file_path: Path to the CSV file to inspect.
    
    Returns:
        String summary listing datetime columns and detected format inconsistencies.
    """   
    
    common_formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y.%m.%d", "%d.%m.%Y", "%m.%d.%Y"
    ]
    
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        date_columns_found = False
        
        for col in df.columns:
            sample_values = df[col].dropna().astype(str).head(50)  
            if sample_values.empty:
                continue
            
            detected_format = None
            for fmt in common_formats:
                try:
                    parsed = pd.to_datetime(sample_values, format=fmt, errors="raise")
                    detected_format = fmt
                    break
                except Exception:
                    continue
            
            if not detected_format:
                try:
                    parsed = sample_values.apply(lambda x: parse(x, fuzzy=False))
                    detected_format = "mixed/auto-detected"
                except Exception:
                    continue
            
            parsed_all = pd.to_datetime(df[col], format=detected_format if detected_format != "mixed/auto-detected" else None, errors="coerce")
            invalid_count = parsed_all.isnull().sum()
            total = len(df)
            
            if parsed_all.notnull().any():
                date_columns_found = True
                result += f"Column '{col}':\n"
                result += f"Detected format - {detected_format}\n"
                result += f"Invalid values - {invalid_count:,} of {total:,} rows\n\n"
        
        if not date_columns_found:
            result += "No datetime-like columns detected.\n"
        
        return result
    
    except Exception as e:
        return f"Error checking date formats: {str(e)}"


@mcp.tool()
def check_categorical_frequency(file_path: str) -> str:
    """
    Analyzes categorical columns for predominant or very low frequency values.
    
    Use when: identifying rare or overrepresented categories that may indicate data quality issues.
    Examples: 'find dominant or rare categories', 'check for outliers in categorical data', 'analyze frequency distribution'
    
    Args:
        file_path: Path to the CSV file to check for categorical frequencies.
    
    Returns:
        String summary with the most common and least common values in categorical columns.
    """
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        result = f"Categorical Frequency Analysis for file: {file_obj}\n\n"
        issues_found = False
        
        for col in df.select_dtypes(include='object').columns:
            value_counts = df[col].value_counts()
            rare_values = value_counts[value_counts <= 2]  # Consider rare if the frequency is <= 2
            dominant_values = value_counts[value_counts > (len(df) // 2)]  # Consider dominant if > 50% of rows
            
            if not rare_values.empty or not dominant_values.empty:
                issues_found = True
                result += f"Column '{col}' has rare or dominant categories:\n"
                result += f"Rare values: {rare_values.to_dict()}\n"
                result += f"Dominant values: {dominant_values.to_dict()}\n"
        
        if not issues_found:
            result += "No problematic categorical values found.\n"
        
        return result
    
    except Exception as e:
        return f"Error checking categorical frequency: {str(e)}"
    
@mcp.tool()
def check_numeric_format(file_path: str) -> str:
    """
    Analyzes numeric columns to identify formatting issues such as decimal commas,
    thousands separators, spaces, currency symbols, and other non-numeric characters.
    
    Use when: validating numeric data formatting before conversion, identifying data import issues,
    preparing for numeric data cleaning.
    Examples: 'check number formatting', 'detect decimal commas', 'analyze numeric formatting'
    
    Args:
        file_path: Path to the CSV file to be analyzed for numeric formatting issues
    
    Returns:
        String containing a detailed analysis of formatting issues found in each column.
    """
    try:
        file_obj = Path(file_path)
        
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj, dtype=str)  
        
        result = f"Numeric Format Analysis for file: {file_obj}\n"
        result += f"Total rows: {len(df):,}\n"
        result += f"Total columns: {len(df.columns)}\n\n"
        
        issues_found = False
        columns_analyzed = 0
        
        
        patterns = {
            'comma_decimal': r'^\d{1,3}(,\d{3})*,\d+$|^\d+,\d+$',  # 1.234,56 OR 123,45
            'period_thousands': r'^\d{1,3}(\.\d{3})*$',  # 1.234.567 
            'mixed_separators': r'.*[,.].*[,.].*',  # Comma AND dot
            'currency_symbols': r'.*[\$€£¥₹₽¢₱₨₦₴₸₺₪₫₵₢₯₲₴₹].*',  # Currency symbols
            'percentage': r'.*%.*',  # Percentage symbols
            'spaces_in_number': r'^\d+(\s\d+)+$',  # Numbers with spaces: 1 234 567
            'parentheses_negative': r'^\(\d+([.,]\d+)?\)$',  # Negative numbers in parentheses
            'leading_zeros': r'^0+\d+$',  # Leading zeros
            'scientific_notation': r'^\d+[eE][+-]?\d+$',  # Scientific notation
            'mixed_characters': r'.*[a-zA-Z].*\d.*|.*\d.*[a-zA-Z].*'  # Letters AND numbers
        }
        
        for col in df.columns:
            col_data = df[col].dropna().astype(str)
            
            if len(col_data) == 0:
                continue
                
            numeric_like_count = 0
            format_issues = {}
            sample_issues = {}
            
            for value in col_data:
                value = value.strip()
                if value == '' or value.lower() in ['nan', 'null', 'none']:
                    continue
                    
                try:
                    float(value)
                    numeric_like_count += 1
                    continue
                except ValueError:
                    pass
                
                for pattern_name, pattern in patterns.items():
                    if re.match(pattern, value):
                        if pattern_name not in format_issues:
                            format_issues[pattern_name] = 0
                            sample_issues[pattern_name] = []
                        format_issues[pattern_name] += 1
                        if len(sample_issues[pattern_name]) < 5:
                            sample_issues[pattern_name].append(value)
                        numeric_like_count += 1
                        break

            # If the column has at least 50% numeric-like values, analyze it
            if numeric_like_count >= len(col_data) * 0.5:
                columns_analyzed += 1
                
                if format_issues:
                    issues_found = True
                    result += f"Column '{col}' - Formatting issues detected:\n"
                    result += f"  Total values analyzed: {len(col_data):,}\n"
                    result += f"  Numeric-like values: {numeric_like_count:,}\n"
                    result += f"  Clean numeric values: {numeric_like_count - sum(format_issues.values()):,}\n\n"
                    
                    for issue_type, count in format_issues.items():
                        percentage = (count / len(col_data)) * 100
                        result += f"  {issue_type.replace('_', ' ').title()}: {count:,} values ({percentage:.1f}%)\n"
                        
                        # Add issue description
                        descriptions = {
                            'comma_decimal': 'Uses comma as decimal separator (European format)',
                            'period_thousands': 'Uses period as thousands separator',
                            'mixed_separators': 'Contains both commas and periods',
                            'currency_symbols': 'Contains currency symbols',
                            'percentage': 'Contains percentage symbols',
                            'spaces_in_number': 'Contains spaces within numbers',
                            'parentheses_negative': 'Uses parentheses for negative numbers',
                            'leading_zeros': 'Has unnecessary leading zeros',
                            'scientific_notation': 'Uses scientific notation',
                            'mixed_characters': 'Contains mix of letters and numbers'
                        }
                        
                        if issue_type in descriptions:
                            result += f"    Description: {descriptions[issue_type]}\n"
                        
                        if sample_issues[issue_type]:
                            result += f"    Examples: {', '.join(sample_issues[issue_type][:3])}\n"

                        # Suggestions for correction
                        suggestions = {
                            'comma_decimal': 'Replace commas with periods: value.replace(",", ".")',
                            'period_thousands': 'Remove periods used as thousands separators',
                            'mixed_separators': 'Standardize to use period as decimal separator',
                            'currency_symbols': 'Remove currency symbols using regex',
                            'percentage': 'Remove % symbol and divide by 100',
                            'spaces_in_number': 'Remove spaces: value.replace(" ", "")',
                            'parentheses_negative': 'Convert to negative numbers with minus sign',
                            'leading_zeros': 'Remove leading zeros: value.lstrip("0")',
                            'scientific_notation': 'Convert using float() if needed',
                            'mixed_characters': 'Extract numeric portion or clean manually'
                        }
                        
                        if issue_type in suggestions:
                            result += f"    Suggestion: {suggestions[issue_type]}\n"
                        
                        result += "\n"
                    
                    result += "  " + "="*50 + "\n\n"
                else:
                    result += f"Column '{col}': Clean numeric formatting ✓\n"
                    result += f"  All {numeric_like_count:,} numeric values are properly formatted\n\n"
        
        result += "="*60 + "\n"
        result += "FORMATTING ANALYSIS SUMMARY:\n"
        result += f"Columns analyzed: {columns_analyzed}\n"
        
        if issues_found:
            result += "Formatting issues found\n\n"
            result += "RECOMMENDATIONS:\n"
            result += "1. Standardize decimal separators (use periods)\n"
            result += "2. Remove currency symbols and percentage signs\n"
            result += "3. Clean thousands separators\n"
            result += "4. Handle negative number formats consistently\n"
            result += "5. Remove unnecessary spaces and leading zeros\n"
            result += "\nConsider using the standardize_numeric_formats() function for automated cleaning.\n"
        else:
            if columns_analyzed > 0:
                result += "All numeric columns have clean formatting\n"
            else:
                result += " No numeric columns detected for analysis\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing numeric formats: {str(e)}"


@mcp.tool()
def check_date_outliers(file_path: str) -> str:
    """
    Detects outliers in date columns by identifying very old dates, far future dates,
    or dates that do not make sense in the context of the data (e.g., people born in the future).

    Use when: validating temporal consistency of data, identifying date entry errors,
    detecting impossible or unlikely dates in context.
    Examples: 'detect date outliers', 'find impossible dates', 'validate historical dates'

    Args:
        file_path: Path to the CSV file to be analyzed for date outliers

    Returns:
        String containing detailed analysis of date outliers found in each column.
    """
    try:
        
        warnings.filterwarnings('ignore')
        
        file_obj = Path(file_path)
        
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        current_date = datetime.now()
        
        result = f"Date Outliers Analysis for file: {file_obj}\n"
        result += f"Analysis date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"Total rows: {len(df):,}\n\n"
        
        date_columns_found = 0
        total_outliers = 0
        columns_with_outliers = 0
        
        reasonable_limits = {
            'very_old': datetime(1900, 1, 1),
            'old': datetime(1950, 1, 1),
            'near_future': current_date + timedelta(days=365),
            'far_future': current_date + timedelta(days=365*10)
        }
        
        for col in df.columns:
            try:
                sample_values = df[col].dropna().astype(str).head(10)
                date_like_count = 0
                
                for val in sample_values:
                    try:
                        pd.to_datetime(val, errors='raise')
                        date_like_count += 1
                    except:
                        continue
                
                # If less than 50% of samples appear to be dates, skip
                if date_like_count < len(sample_values) * 0.5:
                    continue
                
                date_series = pd.to_datetime(df[col], errors='coerce')
                valid_dates = date_series.dropna()
                
                if len(valid_dates) == 0:
                    continue
                
                date_columns_found += 1
                
                outliers_found = {}
                
                very_old = valid_dates[valid_dates < reasonable_limits['very_old']]
                if not very_old.empty:
                    outliers_found['very_old'] = very_old
                
                far_future = valid_dates[valid_dates > reasonable_limits['far_future']]
                if not far_future.empty:
                    outliers_found['far_future'] = far_future
                
                col_lower = col.lower()
                
                # To birth dates
                if any(word in col_lower for word in ['birth', 'born', 'nasc', 'idade', 'age']):
                    future_births = valid_dates[valid_dates > current_date]
                    if not future_births.empty:
                        outliers_found['future_births'] = future_births
                    
                    very_old_people = valid_dates[valid_dates < (current_date - timedelta(days=365*120))]
                    if not very_old_people.empty:
                        outliers_found['very_old_people'] = very_old_people

                # To transaction dates
                elif any(word in col_lower for word in ['event', 'transaction', 'sale', 'purchase', 'order', 'venda', 'compra']):
                    far_future_events = valid_dates[valid_dates > reasonable_limits['near_future']]
                    if not far_future_events.empty:
                        outliers_found['far_future_events'] = far_future_events
                
                # 4. Statistical detection using IQR on timestamps
                timestamps = valid_dates.astype('int64') // 10**9  # Convert to seconds
                Q1 = timestamps.quantile(0.25)
                Q3 = timestamps.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  
                    lower_bound = Q1 - 3 * IQR  
                    upper_bound = Q3 + 3 * IQR
                    
                    statistical_outliers = valid_dates[
                        (timestamps < lower_bound) | (timestamps > upper_bound)
                    ]
                    if not statistical_outliers.empty:
                        outliers_found['statistical'] = statistical_outliers
                
                # Reportar resultados
                if outliers_found:
                    columns_with_outliers += 1
                    column_outlier_count = sum(len(outliers) for outliers in outliers_found.values())
                    total_outliers += column_outlier_count
                    
                    result += f"Column '{col}' - {column_outlier_count:,} date outliers found:\n"
                    result += f"  Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}\n"
                    result += f"  Valid dates: {len(valid_dates):,}\n"
                    result += f"  Invalid dates: {len(df[col]) - len(valid_dates):,}\n\n"
                    
                    for outlier_type, outlier_dates in outliers_found.items():
                        percentage = (len(outlier_dates) / len(valid_dates)) * 100
                        result += f"  {outlier_type.replace('_', ' ').title()}: {len(outlier_dates):,} dates ({percentage:.2f}%)\n"
                        
                        
                        descriptions = {
                            'very_old': f'Dates before {reasonable_limits["very_old"].year} (possibly data entry errors)',
                            'far_future': f'Dates more than 10 years in the future (may be unrealistic)',
                            'future_births': 'Birth dates in the future (impossible)',
                            'very_old_people': 'Birth dates indicating age > 120 years (unlikely)',
                            'far_future_events': 'Events scheduled too far in the future (may be errors)',
                            'statistical': 'Dates that are statistical outliers using IQR method'
                        }
                        
                        if outlier_type in descriptions:
                            result += f"    Issue: {descriptions[outlier_type]}\n"
                        
                        # Show examples
                        sorted_outliers = outlier_dates.sort_values()
                        if len(outlier_dates) <= 5:
                            examples = [d.strftime('%Y-%m-%d') for d in sorted_outliers]
                            result += f"    Examples: {', '.join(examples)}\n"
                        else:
                            first_3 = [d.strftime('%Y-%m-%d') for d in sorted_outliers.head(3)]
                            last_2 = [d.strftime('%Y-%m-%d') for d in sorted_outliers.tail(2)]
                            result += f"    Examples: {', '.join(first_3)} ... {', '.join(last_2)}\n"
                        
                        result += "\n"
                    
                    result += "  " + "="*50 + "\n\n"
                
                else:
                    result += f"Column '{col}': No date outliers detected\n"
                    result += f"  Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}\n"
                    result += f"  All {len(valid_dates):,} dates appear reasonable\n\n"
            
            except Exception as e:
                continue  # Skip columns that can't be converted to dates
        
        # Summary
        result += "="*60 + "\n"
        result += "DATE OUTLIERS SUMMARY:\n"
        result += f"Date columns analyzed: {date_columns_found}\n"
        result += f"Total date outliers found: {total_outliers:,}\n"
        result += f"Columns with outliers: {columns_with_outliers}/{date_columns_found}\n"
        
        if date_columns_found == 0:
            result += "No date columns detected in the dataset\n"
        elif total_outliers == 0:
            result += "No date outliers detected\n"
        else:
            result += "Status: Date outliers found\n\n"
            result += "RECOMMENDATIONS:\n"
            result += "1. Verify outlier dates with source data\n"
            result += "2. Check for data entry errors (wrong century, typos)\n"
            result += "3. Consider if future dates are intentional (scheduled events)\n"
            result += "4. Validate business logic constraints\n"
            result += "5. Consider removing or correcting impossible dates\n"
            result += "6. For birth dates, verify against reasonable human lifespan\n"
        
        return result
        
    except Exception as e:
        return f"Error detecting date outliers: {str(e)}"

@mcp.tool()
def check_category_standardization(file_path: str, category_mapping: dict) -> str:
    """
    Identifies potential categorical mistyping errors and standardizes values based on a provided dictionary.
    
    Use when: standardizing categorical variables that may contain different spellings or abbreviations.
    Examples: 'standardize categories', 'correct category mistyping', 'apply category mapping'
    
    Args:
        file_path: Path to the CSV file to check for category standardization.
        category_mapping: Dictionary to standardize category values. For example: {'Heinken': 'Heineken', 'HNK': 'Heineken'}
    
    Returns:
        String summary with detected mistyped categories and their corrections.
    """
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        result = f"Category Standardization Analysis for file: {file_obj}\n\n"
        issues_found = False
        
        for col in df.select_dtypes(include='object').columns:
            mistyped_categories = df[col].apply(lambda x: category_mapping.get(x, x))
            corrections = mistyped_categories[mistyped_categories != df[col]].value_counts()
            
            if not corrections.empty:
                issues_found = True
                result += f"Column '{col}' has category standardization issues:\n"
                result += f"Corrections required: {corrections.to_dict()}\n"
        
        if not issues_found:
            result += "No category standardization issues found.\n"
        
        return result
    
    except Exception as e:
        return f"Error checking category standardization: {str(e)}"
    
@mcp.tool()
def check_primary_keys(file_path: str) -> str:
    """
    Identifies columns that could serve as primary keys by checking for uniqueness and nulls.
    
    Use when: verifying potential primary keys for database design or data integrity.
    Examples: 'find candidate primary keys', 'check for unique identifiers', 'validate potential keys'
    
    Args:
        file_path: Path to the CSV file to check for potential primary keys.
    
    Returns:
        String summary with columns that are unique and could serve as primary keys.
    """
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        result = f"Primary Key Analysis for file: {file_obj}\n\n"
        possible_keys = []
        
        for col in df.columns:
            if df[col].is_unique and df[col].notna().all():
                possible_keys.append(col)
        
        if possible_keys:
            result += f"Possible primary key columns: {', '.join(possible_keys)}\n"
        else:
            result += "No suitable primary key columns found.\n"
        
        return result
    
    except Exception as e:
        return f"Error checking primary keys: {str(e)}"

@mcp.tool()
def propose_table_name(file_path: str) -> str:
    """
    Proposes a business-friendly name for the table based on its content.
    
    Use when: assigning a meaningful name to a dataset based on its columns or data content.
    Examples: 'suggest a table name', 'propose a business-friendly name for this data', 'name this table based on contents'
    
    Args:
        file_path: Path to the CSV file for which a table name is to be proposed.
    
    Returns:
        String with a proposed business-friendly name for the table.
    """
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        # Generate a table name suggestion based on the most common column names or the dataset content
        name_suggestion = "table_" + "_".join(df.columns.str.lower())
        
        result = f"Proposed table name based on content: {name_suggestion}\n"
        
        return result
    
    except Exception as e:
        return f"Error proposing table name: {str(e)}"

@mcp.tool()
def check_outliers_numbers(file_path: str) -> str:
    """
    Detects outliers in numeric columns using the IQR (Interquartile Range) method.
    
    The IQR method considers outliers as values that are:
    - Below Q1 - 1.5 * IQR (lower bound)
    - Above Q3 + 1.5 * IQR (upper bound)
    
    Use when: identifying extreme values in numeric data, detecting anomalies, preparing data for analysis.
    Examples: 'detect outliers', 'find extreme values', 'analyze anomalies in numbers'
    
    Args:
        file_path: Path to the CSV file to be analyzed for outliers
    
    Returns:
        String containing detailed analysis of outliers found in each numeric column.
    """
    try:
        file_obj = Path(file_path)
        
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        # Selecionar apenas colunas numéricas
        numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        
        if len(numeric_columns) == 0:
            return f"No numeric columns found in {file_obj}"
        
        result = f"Outlier Analysis (IQR Method) for file: {file_obj}\n"
        result += f"Total rows: {len(df):,}\n"
        result += f"Numeric columns analyzed: {len(numeric_columns)}\n\n"
        
        total_outliers = 0
        columns_with_outliers = 0
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                result += f"Column '{col}': No data available (all null values)\n\n"
                continue
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                columns_with_outliers += 1
                total_outliers += outlier_count
                
                result += f"Column '{col}' - {outlier_count:,} outliers found:\n"
                result += f"  Q1: {Q1:.2f}\n"
                result += f"  Q3: {Q3:.2f}\n"
                result += f"  IQR: {IQR:.2f}\n"
                result += f"  Lower bound: {lower_bound:.2f}\n"
                result += f"  Upper bound: {upper_bound:.2f}\n"
                result += f"  Outlier percentage: {(outlier_count / len(col_data)) * 100:.2f}%\n"
                
                result += f"  Min outlier: {outliers.min():.2f}\n"
                result += f"  Max outlier: {outliers.max():.2f}\n"
                
                if outlier_count <= 10:
                    result += f"  Outlier values: {', '.join([f'{x:.2f}' for x in sorted(outliers)])}\n"
                else:
                    sorted_outliers = sorted(outliers)
                    result += f"  First 5 outliers: {', '.join([f'{x:.2f}' for x in sorted_outliers[:5]])}\n"
                    result += f"  Last 5 outliers: {', '.join([f'{x:.2f}' for x in sorted_outliers[-5:]])}\n"
                
                result += "\n"
            else:
                result += f"Column '{col}': No outliers found\n"
                result += f"  Range: {col_data.min():.2f} to {col_data.max():.2f}\n"
                result += f"  IQR: {IQR:.2f}\n\n"
        
        # Resumo geral
        result += "=" * 60 + "\n"
        result += "SUMMARY:\n"
        result += f"Total outliers found: {total_outliers:,}\n"
        result += f"Columns with outliers: {columns_with_outliers}/{len(numeric_columns)}\n"
        
        if total_outliers > 0:
            result += f"Overall outlier rate: {(total_outliers / (len(df) * len(numeric_columns))) * 100:.2f}%\n"
            result += "\nRecommendation: Review outliers to determine if they are:\n"
            result += "- Data entry errors that need correction\n"
            result += "- Valid extreme values that should be kept\n"
            result += "- Values that need transformation or special handling\n"
        else:
            result += "No outliers detected in any numeric column.\n"
        
        return result
        
    except Exception as e:
        return f"Error detecting outliers: {str(e)}" 

@mcp.tool()
def remove_duplicates(file_path: str) -> str:
    """
    Removes all duplicate rows from the DataFrame.
    
    Use when: cleaning data by removing repeated rows.
    Examples: 'remove duplicates', 'clean data', 'eliminate duplicate rows'
    
    Args:
        file_path: Path to the CSV file to remove duplicates.
    
    Returns:
        String indicating the number of duplicates removed and the cleaned file.
    """
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        # Remove duplicates
        before_count = len(df)
        df_cleaned = df.drop_duplicates()
        after_count = len(df_cleaned)
        
        # Save the cleaned file
        cleaned_file_path = file_path.replace(".csv", "_cleaned.csv")
        df_cleaned.to_csv(cleaned_file_path, index=False)
        
        result = f"Removed {before_count - after_count} duplicate rows.\n"
        result += f"Cleaned file saved as {cleaned_file_path}."
        
        return result
    
    except Exception as e:
        return f"Error removing duplicates: {str(e)}"

@mcp.tool()
def fill_missing_values_zero(file_path: str) -> str:
    """
    Fills missing values in the DataFrame with zero.
    
    Use when: preparing data for analysis where missing values should be treated as zero.
    Examples: 'fill missing values with zero', 'replace nulls with 0', 'clean missing data'
    
    Args:
        file_path: Path to the CSV file to fill missing values.
    
    Returns:
        String indicating how many missing values were filled and the cleaned file.
    """
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        # Fill missing values with zero
        before_count = df.isnull().sum().sum()
        df_filled = df.fillna(0)
        after_count = df_filled.isnull().sum().sum()
        
        # Save the cleaned file
        filled_file_path = file_path.replace(".csv", "_filled.csv")
        df_filled.to_csv(filled_file_path, index=False)
        
        result = f"Filled {before_count:,} missing values with zero.\n"
        result += f"Cleaned file saved as {filled_file_path}."
        
        return result
    
    except Exception as e:
        return f"Error filling missing values with zero: {str(e)}"


@mcp.tool()
def standardize_column_names(file_path: str) -> str:
    """
    Standardizes column names to snake_case and removes unnecessary spaces.
    
    Use when: ensuring consistent column naming convention.
    Examples: 'standardize column names', 'convert columns to snake_case', 'clean column headers'
    
    Args:
        file_path: Path to the CSV file to clean.
    
    Returns:
        String indicating the new column names and the cleaned file.
    """
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        # Standardize column names to snake_case and strip spaces
        new_columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        df.columns = new_columns
        
        # Save the cleaned file
        cleaned_file_path = file_path.replace(".csv", "_standardized_columns.csv")
        df.to_csv(cleaned_file_path, index=False)
        
        result = f"Column names standardized to snake_case: {', '.join(new_columns)}.\n"
        result += f"Cleaned file saved as {cleaned_file_path}."
        
        return result
    
    except Exception as e:
        return f"Error standardizing column names: {str(e)}"

@mcp.tool()
def standardize_numeric_formats(file_path: str) -> str:
    """
    Standardizes numeric formats by replacing commas with periods and ensuring consistency.
    
    Use when: cleaning numeric data with inconsistent formats (e.g., commas as decimal separators).
    Examples: 'standardize numeric formats', 'convert commas to periods in numbers', 'clean numeric columns'
    
    Args:
        file_path: Path to the CSV file to clean.
    
    Returns:
        String indicating which numeric columns were cleaned and the cleaned file.
    """
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        # Replace commas with periods in numeric columns
        numeric_columns = df.select_dtypes(include='number').columns
        for col in numeric_columns:
            df[col] = df[col].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
        
        # Save the cleaned file
        cleaned_file_path = file_path.replace(".csv", "_standardized_numeric.csv")
        df.to_csv(cleaned_file_path, index=False)
        
        result = f"Standardized numeric formats in {len(numeric_columns)} columns.\n"
        result += f"Cleaned file saved as {cleaned_file_path}."
        
        return result
    
    except Exception as e:
        return f"Error standardizing numeric formats: {str(e)}"


@mcp.tool()
def standardize_date_formats(file_path: str) -> str:
    """
    Standardizes date columns to the 'yyyy-mm-dd' format.
    
    Use when: ensuring consistent date format across the dataset.
    Examples: 'standardize date formats', 'convert dates to yyyy-mm-dd', 'clean date columns'
    
    Args:
        file_path: Path to the CSV file to clean.
    
    Returns:
        String indicating how many date columns were standardized and the cleaned file.
    """
      
    common_formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y.%m.%d", "%d.%m.%Y", "%m.%d.%Y"
    ]
    
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        standardized_columns = []
        
        for col in df.columns:
            if df[col].dtype == object or "date" in col.lower():
                sample_values = df[col].dropna().astype(str).head(30)
                if sample_values.empty:
                    continue
                
                detected_format = None
                for fmt in common_formats:
                    try:
                        pd.to_datetime(sample_values, format=fmt, errors="raise")
                        detected_format = fmt
                        break
                    except Exception:
                        continue
                
                if detected_format:
                    df[col] = pd.to_datetime(df[col], format=detected_format, errors="coerce").dt.strftime("%Y-%m-%d")
                    standardized_columns.append(col)
                else:
                    try:
                        df[col] = df[col].apply(
                            lambda x: parse(str(x)).strftime("%Y-%m-%d") if pd.notna(x) else None
                        )
                        standardized_columns.append(col)
                    except Exception:
                        continue
        
        cleaned_file_path = file_obj.with_name(file_obj.stem + "_standardized_dates.csv")
        df.to_csv(cleaned_file_path, index=False)
        
        if standardized_columns:
            result = f"Standardized date formats in {len(standardized_columns)} column(s): {', '.join(standardized_columns)}\n"
            result += f"Cleaned file saved as: {cleaned_file_path}"
        else:
            result = "No date-like columns were standardized."
        
        return result
    
    except Exception as e:
        return f"Error standardizing date formats: {str(e)}"

# Run the server 
if __name__ == "__main__":
    mcp.run()