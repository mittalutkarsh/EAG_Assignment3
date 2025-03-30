# Finance Agent Chrome Extension

A Chrome extension powered by Gemini AI that generates, backtests, and evaluates algorithmic trading strategies for Bitcoin.

![Finance Agent Screenshot](screenshots/finance_agent_screenshot.png)

## Features

- **AI-Powered Strategy Generation**: Create original trading strategies using Gemini AI
- **Backtesting Engine**: Test strategies against historical Bitcoin price data
- **Strategy Evaluation**: Get AI feedback on strategy performance and suggestions for improvement
- **LLM Interaction Logs**: View and export logs of all AI interactions

## Installation

### Prerequisites

- Python 3.7+
- Google Chrome browser
- Gemini API key

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/finance-agent-extension.git
   cd finance-agent-extension
   ```

2. Install Python dependencies:
   ```bash
   pip install flask flask-cors pandas numpy requests matplotlib google-generativeai python-dotenv
   ```

3. Create a `.env` file in the project root and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" using the toggle in the top-right corner
3. Click "Load unpacked"
4. Browse to and select the `chrome_extension` folder within the project directory

## Usage

1. Start the Flask backend server:
   ```bash
   python finance_api.py
   ```

2. Click the Finance Agent icon in your Chrome toolbar to open the extension

3. Generate a trading strategy:
   - Enter a description of the strategy you want to create
   - Click "Generate Strategy"

4. Backtest the strategy:
   - Set the number of days for backtesting
   - Click "Run Backtest"
   - View backtest results

5. Evaluate the strategy:
   - After backtesting, click "Evaluate Strategy"
   - View the AI's evaluation and suggestions

6. View and copy LLM logs:
   - Switch to the "LLM Logs" tab
   - Click "Refresh Logs" to update
   - Click "Copy All Logs" to copy logs to clipboard

## Project Structure

```
finance-agent-extension/
├── EAG_Assignment3/
│   ├── finance_agent.py     # Core AI and backtesting functionality
│   ├── finance_api.py       # Flask API server
│   └── chrome_extension/    # Chrome extension files
│       ├── manifest.json    # Extension configuration
│       ├── popup.html       # Extension UI
│       ├── popup.js         # Extension logic
│       └── images/          # Extension icons
├── .env                     # Environment variables (not in Git)
└── README.md                # This file
```

## Technologies Used

- **Backend**:
  - Python 
  - Flask (web server)
  - Pandas & NumPy (data processing)
  - Google Generative AI (Gemini 2.0)

- **Frontend**:
  - HTML/CSS
  - JavaScript
  - Chrome Extension API

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [CoinGecko API](https://www.coingecko.com/en/api) for Bitcoin price data
- [Google Gemini AI](https://ai.google.dev/) for strategy generation and evaluation 