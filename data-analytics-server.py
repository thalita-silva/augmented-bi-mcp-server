"""
Enhanced CSV Data Analytics MCP Server

This server provides tools for reading, analyzing, and performing data quality checks on CSV files via FastMCP.

Available tools:
- read_csv: Preview CSV contents and structure.
- aggregate_csv: Group and aggregate data by columns.
- find_missing_values: Identify columns with missing data.
- detect_duplicates: Analyze duplicate rows.

Example usage:
- /read_csv file_path="sample.csv"
- /aggregate_csv file_path="sample.csv" group_by="Category" agg_column="Sales_Amount" agg_function="sum"
- /aggregate_csv file_path="sample.csv" group_by="Region,Customer_Type" agg_column="Units_Sold" agg_function="mean"
- /find_missing_values file_path="sample.csv"
- /detect_duplicates file_path="sample.csv"
"""

from fastmcp import FastMCP
import pandas as pd
from pathlib import Path

mcp = FastMCP("data-analytics-server")

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
def aggregate_csv(file_path: str, group_by: str, agg_column: str, agg_function: str) -> str:
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
def find_missing_values(file_path: str) -> str:
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
def detect_duplicates(file_path: str) -> str:
    """
    Checks a CSV file for duplicate rows and provides detailed duplicate analysis.
    
    Use when: validating data uniqueness, identifying data entry errors, cleaning datasets.
    Examples: 'check for duplicate rows', 'find repeated records', 'analyze data duplicates'
    
    Args:
        file_path: Path to the CSV file to check for duplicates
    
    Returns:
        String containing duplicate analysis including total count and affected rows.
    """
    try:
        file_obj = Path(file_path)
        
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        duplicated_rows = df.duplicated()
        duplicate_count = duplicated_rows.sum()
        
        result = f"Duplicate Rows Analysis:\n"
        result += f"File: {file_obj}\n"
        result += f"Total rows: {len(df):,}\n\n"
        
        if duplicate_count == 0:
            result += "No duplicate rows found!\n"
        else:
            result += f"Found {duplicate_count:,} duplicate row(s)\n\n"
            
            duplicate_indices = df[duplicated_rows].index.tolist()
            result += f"Duplicate row indices: {duplicate_indices}\n\n"
            
            if duplicate_count <= 10: 
                result += "Duplicate rows:\n"
                result += df[duplicated_rows].to_string()
            else:
                result += f"Too many duplicates to display ({duplicate_count} rows)\n"
                result += "First 5 duplicate rows:\n"
                result += df[duplicated_rows].head().to_string()
            
            duplicate_percentage = (duplicate_count / len(df)) * 100
            result += f"\n\nDuplicate percentage: {duplicate_percentage:.2f}%\n"
            
            unique_rows = len(df) - duplicate_count
            result += f"Unique rows: {unique_rows:,}\n"
        
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
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        result = f"Date Format Analysis for file: {file_obj}\n\n"
        date_columns_found = False
        
        for col in df.columns:
            try:
                # Attempt to convert column to datetime
                parsed_dates = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                if parsed_dates.notnull().any():
                    date_columns_found = True
                    total = len(df)
                    invalid = parsed_dates.isnull().sum()
                    result += f"Column '{col}': {invalid:,} invalid date(s) out of {total:,} rows\n"
            except Exception:
                continue
        
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
def fix_encoding_and_characters(file_path: str) -> str:
    """
    Fixes encoding issues and removes special characters from string columns.
    
    Use when: cleaning text data by fixing encoding problems and removing unwanted characters.
    Examples: 'fix encoding and special characters', 'clean text data', 'standardize text columns'
    
    Args:
        file_path: Path to the CSV file to clean.
    
    Returns:
        String indicating how many text columns were cleaned and the cleaned file.
    """
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj, encoding='utf-8', errors='replace')
        
        # Remove special characters from text columns
        text_columns = df.select_dtypes(include='object').columns
        for col in text_columns:
            df[col] = df[col].str.replace(r'[^\x00-\x7F]+', '', regex=True)  # Remove non-ASCII characters
        
        # Save the cleaned file
        cleaned_file_path = file_path.replace(".csv", "_cleaned_encoding.csv")
        df.to_csv(cleaned_file_path, index=False)
        
        result = f"Fixed encoding issues and removed special characters in {len(text_columns)} text columns.\n"
        result += f"Cleaned file saved as {cleaned_file_path}."
        
        return result
    
    except Exception as e:
        return f"Error fixing encoding and special characters: {str(e)}"

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
    try:
        file_obj = Path(file_path)
        if not file_obj.exists():
            return f"Error: File not found at {file_obj}"
        
        df = pd.read_csv(file_obj)
        
        # Standardize date columns to 'yyyy-mm-dd'
        date_columns = df.select_dtypes(include='datetime').columns
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Save the cleaned file
        cleaned_file_path = file_path.replace(".csv", "_standardized_dates.csv")
        df.to_csv(cleaned_file_path, index=False)
        
        result = f"Standardized date formats in {len(date_columns)} columns.\n"
        result += f"Cleaned file saved as {cleaned_file_path}."
        
        return result
    
    except Exception as e:
        return f"Error standardizing date formats: {str(e)}"

# Run the server 
if __name__ == "__main__":
    mcp.run()