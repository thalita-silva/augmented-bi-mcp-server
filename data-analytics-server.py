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

# Run the server 
if __name__ == "__main__":
    mcp.run()