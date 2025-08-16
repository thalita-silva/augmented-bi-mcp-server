# [From Chaos to Cube] ETL Agent | Augmented BI to data cleaning using a MCP Server

Handling raw data with inconsistencies, varying formats, and duplicates is a repetitive and time-consuming task for analysts. Our challenge is to accelerate this step with a smart, accessible solution with an interactive interface that not only executes commands but also automatically identifies improvements and issues in the databases, proactively proposing corrections.


## 🚀 Features

### Data Analysis Tools
- **CSV Reading & Preview** - View file structure and sample data
- **Data Aggregation** - Group and aggregate data with various functions
- **Data Quality Checks** - Comprehensive validation and quality assessment

### Data Quality Assessment
- ✅ Missing value detection and analysis
- ✅ Duplicate row identification
- ✅ Data type consistency checking
- ✅ Date format validation and outlier detection
- ✅ Categorical frequency analysis
- ✅ Numeric format standardization
- ✅ Primary key candidate identification
- ✅ Outlier detection using IQR method

### Data Cleaning & Standardization
- 🧹 Remove duplicate rows
- 🧹 Fill missing values with zero
- 🧹 Standardize column names to snake_case
- 🧹 Standardize numeric formats (comma to period conversion)
- 🧹 Standardize date formats to yyyy-mm-dd
- 🧹 Category standardization with custom mappings

## 📋 Prerequisites

- Python 3.8 or higher
- UV package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

## 🛠️ Installation

1. **Clone or download the script** to your local directory
2. **Ensure you have UV installed**:
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install dependencies** (UV will handle this automatically when you run the script):
   ```bash
   uv add fastmcp pandas python-dateutil
   ```

## 🚀 Usage

### Running the Server

```bash
uv run from-chaos-to-cube-server.py
```

## 🔧 Technical Details

- **Framework:** FastMCP (Model Context Protocol)
- **Data Processing:** Pandas
- **Date Parsing:** python-dateutil
- **Statistical Methods:** IQR for outlier detection
- **Format Support:** CSV files 

## 📊 Performance Considerations

- Optimized for files up to several million rows
- Memory-efficient processing with pandas
- Automatic data type inference
- Streaming support for large datasets
