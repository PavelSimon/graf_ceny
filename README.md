# Electricity Price & Consumption Analyzer

This Streamlit web app helps you analyze electricity prices and your consumption costs by combining data from a local SQLite database and your own Excel file.

## Features
- Upload your electricity consumption data (Excel, kWh)
- Reads electricity price data from a local SQLite database (`dam_data.db`)
- Combines both datasets to calculate and visualize:
  - Electricity price over time
  - Total consumption cost over time
- Interactive date range selection (defaulted to your Excel data, but user-adjustable)

## Setup
1. Clone this repository or download the files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your `dam_data.db` (SQLite) and your Excel file in the project directory.

## Usage
Run the app with:
```bash
streamlit run app.py
```

- Upload your Excel file when prompted.
- Adjust the date range as needed.
- View the interactive graphs.

## Requirements
- Python 3.7+
- `dam_data.db` (SQLite database with electricity prices)
- Your consumption Excel file (with date and kWh columns)

---

Feel free to customize the app for your own data formats! 