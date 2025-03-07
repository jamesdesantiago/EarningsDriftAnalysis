# Earnings Drift Analyzer

## Overview
Earnings Drift Analyzer is a Dash web application that visualizes stock earnings drift analysis using historical earnings data and index movements. The application fetches earnings data from Google Sheets and index data from Yahoo Finance to create insightful visualizations and filter earnings trends based on various metrics.

Note: Without properly formatting your google sheets document and processin earnings data beforehand, you will will struggle to use the script. There's a significan amount of work that goes into processing the earnings data and building a database. This script only helps to display this data in a user friendly way.

## Features
- Fetch earnings data from a Google Sheet.
- Retrieve historical stock index data from Yahoo Finance.
- Interactive filters to analyze stock earnings drift.
- Bubble chart visualization of positive, negative, and net earnings drift.
- Select and display earnings data for specific dates.
- Download filtered earnings data as a CSV file.

## Installation
To run the application locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Earnings-Drift-Analyzer.git
   cd Earnings-Drift-Analyzer
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set up Google Sheets API credentials:
   - Obtain Google Sheets API credentials and save them as `gsheets_credentials.json` in the `assets` folder.

5. Run the application:
   ```sh
   python app.py
   ```

6. Open your browser and navigate to:
   ```
   http://127.0.0.1:8050/
   ```

## Dependencies
Ensure you have the following dependencies installed (managed via `requirements.txt`):
- `dash`
- `plotly`
- `pandas`
- `numpy`
- `yfinance`
- `gspread`
- `oauth2client`

## File Structure
```
Earnings-Drift-Analyzer/
│-- app.py                   # Main Dash application
│-- requirements.txt         # Python dependencies
│-- assets/
│   └── gsheets_credentials.json  # Google Sheets API credentials
```

## Usage
- Select an index and year to analyze earnings drift.
- Adjust filter sliders to refine data selection.
- Click on a bubble in the chart to view details for a specific date.
- Download the filtered dataset as a CSV file.

## Contributing
Feel free to fork this repository and submit pull requests for improvements or new features.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or suggestions, please reach out via GitHub Issues.

