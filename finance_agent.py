import asyncio
from dataclasses import dataclass
from typing import Literal
import google.generativeai as genai
import math
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=gemini_api_key)

# Create a model
model = genai.GenerativeModel('gemini-2.0-flash')
#current_query = """Create a moving average crossover strategy for Bitcoin"""

# System prompt configuration

# Function to fetch Bitcoin historical data from CoinGecko

system_prompt = """You are a trading strategy agent creating original code. You have two roles:

STRATEGY_GENERATOR:
- Create original and unique trading strategy implementations in Python
- Design a simple but effective strategy using moving averages for Bitcoin
- Use the provided 'data' DataFrame with date index and 'price' column
- YOUR CODE MUST DEFINE A DATAFRAME NAMED 'results' AT THE END
- The 'results' DataFrame must have these columns:
  * 'price': original price data
  * 'short_ma': short-term moving average
  * 'long_ma': long-term moving average
  * 'signal': buy (1), sell (-1), or hold (0) signals
  * 'position': position size
  * 'returns': daily returns
  * 'strategy_returns': returns from the strategy
  * 'cumulative_returns': cumulative returns from the strategy
- IMPORTANT: Make sure your final DataFrame is named 'results'
- DO NOT just return a function - execute your strategy and store output in 'results'

STRATEGY_EVALUATOR:
- Evaluate the trading strategy implementations
- Determine if the strategy is:
   * successful
   * needs_refinement
   * unsuccessful
- Provide detailed feedback for improvement

CRITICAL: ALWAYS format your response EXACTLY as follows:
1. For strategy generation: STRATEGY_IMPLEMENTATION: [python_code]
2. For strategy evaluation: STRATEGY_EVALUATION: [feedback]|[score]

The score MUST be one of: successful, needs_refinement, unsuccessful
DO NOT use any other format. DO NOT include multiple responses.
"""

current_query = """Create an original moving average crossover strategy for Bitcoin with a 15-day short MA and 40-day long MA. Include risk management with a 2% stop loss per trade and position sizing based on volatility. Use the provided 'data' DataFrame."""
def fetch_bitcoin_data(days=365):
    print(f"Fetching Bitcoin data for {days} days...")
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prices = data["prices"]
            
            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df[["date", "price"]]
            df.set_index("date", inplace=True)
            
            print(f"Successfully fetched {len(df)} days of Bitcoin data")
            return df
        else:
            print(f"Error fetching data: HTTP {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
            # Create synthetic data as a fallback
            print("Creating synthetic Bitcoin data as fallback")
            dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
            synthetic_data = pd.DataFrame(
                index=dates,
                data={
                    'price': np.linspace(20000, 30000, days) + np.random.normal(0, 1000, days)
                }
            )
            return synthetic_data
    except Exception as e:
        print(f"Exception fetching Bitcoin data: {e}")
        
        # Create synthetic data as a fallback
        print("Creating synthetic Bitcoin data as fallback due to exception")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
        synthetic_data = pd.DataFrame(
            index=dates,
            data={
                'price': np.linspace(20000, 30000, days) + np.random.normal(0, 1000, days)
            }
        )
        return synthetic_data

# Function to backtest a strategy on historical data
def backtest_strategy(strategy_code, data):
    print("\n=== STARTING BACKTEST ===")
    print(f"Data shape: {data.shape}")
    print(f"First few rows of data:\n{data.head()}")
    
    # Create a namespace to execute the strategy code
    namespace = {"data": data.copy(), "np": np, "pd": pd}
    
    # Execute the strategy code in the namespace
    try:
        # Print the first few lines of code for debugging
        print("Executing strategy code (first 10 lines):")
        print("\n".join(strategy_code.split("\n")[:10]) + "...")
        
        # Add syntax validation - this will raise a SyntaxError if the code is invalid
        compile(strategy_code, '<string>', 'exec')
        
        # Add code to ensure we have a 'results' DataFrame at the end
        augmented_code = strategy_code + """
# Code added by backtest_strategy to help locate results
print('Checking for results DataFrame...')

# Check if we don't have a results DataFrame yet
if 'results' not in locals():
    print('No results DataFrame found, searching for alternatives...')
    # Look for other DataFrame variables that might contain strategy results
    candidate_dfs = []
    for var_name, var_val in list(locals().items()):
        if isinstance(var_val, pd.DataFrame) and var_name != 'data':
            # Found a potential results DataFrame
            print(f"Found DataFrame '{var_name}' with shape {var_val.shape}")
            candidate_dfs.append((var_name, var_val))
    
    # If we found any DataFrame candidates, use the largest one
    if candidate_dfs:
        largest_df = max(candidate_dfs, key=lambda x: x[1].shape[0])
        print(f"Using '{largest_df[0]}' as results DataFrame")
        results = largest_df[1]
    else:
        # If we still don't have results, try to construct one from the data
        print('No DataFrames found, attempting to build one from scratch')
        try:
            # Build a basic DataFrame from the original data
            results = data.copy()
            results.columns = ['price']  # Ensure the price column is named correctly
            
            # Calculate short and long MAs if they're in the namespace
            if 'short_window' in locals():
                short_window = locals()['short_window'] 
                results['short_ma'] = results['price'].rolling(window=short_window).mean()
            elif 'short_ma_window' in locals():
                short_window = locals()['short_ma_window']
                results['short_ma'] = results['price'].rolling(window=short_window).mean()
            else:
                # Default to 15-day MA
                results['short_ma'] = results['price'].rolling(window=15).mean()
                
            if 'long_window' in locals():
                long_window = locals()['long_window']
                results['long_ma'] = results['price'].rolling(window=long_window).mean()
            elif 'long_ma_window' in locals():
                long_window = locals()['long_ma_window']
                results['long_ma'] = results['price'].rolling(window=long_window).mean()
            else:
                # Default to 40-day MA
                results['long_ma'] = results['price'].rolling(window=40).mean()
            
            # Create signal column (1 when short_ma > long_ma, -1 otherwise)
            results['signal'] = 0
            results.loc[results['short_ma'] > results['long_ma'], 'signal'] = 1
            results.loc[results['short_ma'] < results['long_ma'], 'signal'] = -1
            
            # Add position column (constant 1 for now)
            results['position'] = 1
            
            # Calculate returns
            results['returns'] = results['price'].pct_change()
            results['strategy_returns'] = results['signal'].shift(1) * results['returns']
            results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
            
            print('Successfully built results DataFrame')
        except Exception as e:
            print(f"Error building results DataFrame: {e}")
"""
        
        # Execute the augmented strategy code
        print("Executing augmented strategy code...")
        exec(augmented_code, namespace)
        
        # Check if the strategy created a 'results' DataFrame
        if "results" in namespace:
            print("Found results DataFrame in namespace")
            results = namespace["results"]
            print(f"Results shape: {results.shape}")
            
            # Ensure the DataFrame has all required columns
            required_columns = ['short_ma', 'long_ma', 'signal', 'strategy_returns']
            missing_columns = [col for col in required_columns if col not in results.columns]
            
            if missing_columns:
                print(f"Warning: Results DataFrame is missing columns: {missing_columns}")
                
                # Try to add missing columns if possible
                if 'short_ma' in missing_columns and 'price' in results.columns:
                    print("Adding short_ma column")
                    results['short_ma'] = results['price'].rolling(window=15).mean()
                
                if 'long_ma' in missing_columns and 'price' in results.columns:
                    print("Adding long_ma column")
                    results['long_ma'] = results['price'].rolling(window=40).mean()
                
                if 'signal' in missing_columns and 'short_ma' in results.columns and 'long_ma' in results.columns:
                    print("Adding signal column")
                    results['signal'] = 0
                    results.loc[results['short_ma'] > results['long_ma'], 'signal'] = 1
                    results.loc[results['short_ma'] < results['long_ma'], 'signal'] = -1
                
                if 'returns' in missing_columns and 'price' in results.columns:
                    print("Adding returns column")
                    results['returns'] = results['price'].pct_change()
                
                if 'strategy_returns' in missing_columns and 'signal' in results.columns and 'returns' in results.columns:
                    print("Adding strategy_returns column")
                    results['strategy_returns'] = results['signal'].shift(1) * results['returns']
                
                if 'cumulative_returns' in missing_columns and 'strategy_returns' in results.columns:
                    print("Adding cumulative_returns column")
                    results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
            
            print(f"Final results columns: {results.columns.tolist()}")
            print(f"Sample of results:\n{results.tail(3)}")
            return results
        else:
            print("No results DataFrame created by strategy")
            
            # Look for any DataFrames in the namespace as a fallback
            for var_name, var_val in namespace.items():
                if isinstance(var_val, pd.DataFrame) and var_name != 'data':
                    print(f"Found DataFrame '{var_name}', using as results")
                    return var_val
            
            print("No DataFrames found in namespace, creating a basic one")
            # Create a simple results DataFrame as a last resort
            results = data.copy()
            results.columns = ['price']  # Ensure columns are named correctly
            
            # Add default strategy components
            results['short_ma'] = results['price'].rolling(window=15).mean()
            results['long_ma'] = results['price'].rolling(window=40).mean()
            results['signal'] = 0
            results.loc[results['short_ma'] > results['long_ma'], 'signal'] = 1
            results.loc[results['short_ma'] < results['long_ma'], 'signal'] = -1
            results['position'] = 1
            results['returns'] = results['price'].pct_change()
            results['strategy_returns'] = results['signal'].shift(1) * results['returns']
            results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
            
            print("Created basic results DataFrame")
            return results
    except Exception as e:
        print(f"Error executing strategy: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        
        # Show the problematic code
        lines = strategy_code.split('\n')
        for i, line in enumerate(lines[:20], 1):  # Show first 20 lines
            print(f"{i}: {line}")
            
        # Try creating a simple DataFrame as a fallback
        try:
            print("Attempting to create fallback DataFrame...")
            results = data.copy()
            results.columns = ['price']
            results['short_ma'] = results['price'].rolling(window=15).mean()
            results['long_ma'] = results['price'].rolling(window=40).mean()
            results['signal'] = 0
            results.loc[results['short_ma'] > results['long_ma'], 'signal'] = 1
            results.loc[results['short_ma'] < results['long_ma'], 'signal'] = -1
            results['position'] = 1
            results['returns'] = results['price'].pct_change()
            results['strategy_returns'] = results['signal'].shift(1) * results['returns']
            results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
            
            print("Successfully created fallback DataFrame")
            return results
        except Exception as fallback_error:
            print(f"Failed to create fallback DataFrame: {fallback_error}")
            return None

# Function to generate a strategy
async def generate_strategy(query):
    prompt = f"{system_prompt}\n\nQuery: {query}"
    response = await model.generate_content_async(prompt)
    return parse_response(response.text)

# Function to evaluate a strategy
async def evaluate_strategy(strategy_code, backtest_results=None):
    try:
        evaluation_query = f"Evaluate the following trading strategy:\n\n{strategy_code}"
        
        if backtest_results is not None:
            # Add backtest results to the evaluation
            # Use a simpler representation for backtest results
            results_summary = backtest_results.tail(10).reset_index().to_string()
            evaluation_query += f"\n\nBacktest Results Summary:\n{results_summary}"
        
        prompt = f"{system_prompt}\n\nQuery: {evaluation_query}"
        print(f"Sending evaluation prompt (first 100 chars): {prompt[:100]}...")
        
        try:
            response = await model.generate_content_async(prompt)
            response_text = response.text
            print(f"RAW LLM RESPONSE START (first 300 chars)\n{response_text[:300]}...\nRAW LLM RESPONSE END")
            
            parsed_response = parse_response(response_text)
            print(f"Parsed response: {parsed_response}")
            
            # Force evaluation type if we got an implementation
            if parsed_response["type"] == "implementation":
                print("Received implementation instead of evaluation, converting to generic evaluation")
                return {
                    "type": "evaluation",
                    "feedback": "The model returned code instead of an evaluation. Please try again.",
                    "score": "needs_refinement"
                }
                
            return parsed_response
        except Exception as model_error:
            print(f"Error generating content: {model_error}")
            return {"type": "error", "message": f"Model error: {str(model_error)}"}
    except Exception as outer_error:
        print(f"Outer evaluation error: {outer_error}")
        return {"type": "error", "message": f"Evaluation error: {str(outer_error)}"}

# Add this function to clean and validate the generated code
def clean_code(code_string):
    """Clean and validate the generated code, fixing common syntax issues."""
    lines = code_string.split('\n')
    fixed_lines = []
    
    # Track quote states
    in_single_quote = False
    in_double_quote = False
    in_triple_single = False
    in_triple_double = False
    
    for i, line in enumerate(lines):
        # Fix unterminated strings by adding closing quotes at the end if needed
        for j, char in enumerate(line):
            if char == "'" and not in_double_quote and not in_triple_double:
                if j > 0 and line[j-1:j+2] == "'''":
                    in_triple_single = not in_triple_single
                else:
                    in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote and not in_triple_single:
                if j > 0 and line[j-1:j+2] == '"""':
                    in_triple_double = not in_triple_double
                else:
                    in_double_quote = not in_double_quote
        
        # If we're in a quote at the end of a line, close it (unless it's a triple quote)
        if in_single_quote and not in_triple_single:
            line += "'"
            in_single_quote = False
        if in_double_quote and not in_triple_double:
            line += '"'
            in_double_quote = False
            
        fixed_lines.append(line)
    
    # Ensure we close any triple quotes at the end of the code
    if in_triple_single:
        fixed_lines.append("'''")
    if in_triple_double:
        fixed_lines.append('"""')
        
    return '\n'.join(fixed_lines)

# Parse response function - improved to be more flexible
def parse_response(response_text):
    # Print for debugging
    print(f"Raw response from LLM (first 200 chars): {response_text[:200]}")
    
    if "STRATEGY_IMPLEMENTATION:" in response_text:
        # Extract the Python code
        code = response_text.split("STRATEGY_IMPLEMENTATION:")[1].strip()
        
        # Remove markdown code formatting if present
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        elif code.startswith("```"):
            code = code[len("```"):].strip()
            
        # Find where the code ends - look for closing backticks or evaluation text
        if "```" in code:
            code = code.split("```")[0].strip()
        if "STRATEGY_EVALUATION:" in code:
            code = code.split("STRATEGY_EVALUATION:")[0].strip()
            
        # Apply code cleaning to fix syntax errors
        code = clean_code(code)
        
        return {"type": "implementation", "code": code}
    elif "STRATEGY_EVALUATION:" in response_text:
        # Extract the feedback and score
        evaluation = response_text.split("STRATEGY_EVALUATION:")[1].strip()
        print(f"Extracted evaluation text: {evaluation}")
        
        # Try to find a score in the format | score
        if "|" in evaluation:
            # Fix for multiple '|' characters - split on the last one
            parts = evaluation.rsplit("|", 1)
            if len(parts) >= 2:
                feedback, score = parts[0], parts[1]
                # Check if there are more pipe characters in the score
                if "|" in score:
                    score = score.split("|")[0]  # Take only the first part
            else:
                feedback = evaluation
                score = "needs_refinement"  # Default fallback
        else:
            # No pipe symbol found, look for keywords in the text
            feedback = evaluation
            score = "needs_refinement"  # Default
            
            if "successful" in evaluation.lower():
                score = "successful"
            elif "unsuccessful" in evaluation.lower():
                score = "unsuccessful"
        
        return {"type": "evaluation", "feedback": feedback.strip(), "score": score.strip()}
    else:
        # Try to guess what this is
        print("No standard format found, attempting to infer response type...")
        
        if "evaluat" in response_text.lower() or "feedback" in response_text.lower():
            # Likely an evaluation without the proper format
            print("Detected evaluation-like content")
            
            # Try to find a score pattern
            score = "needs_refinement"  # Default
            if "successful" in response_text.lower():
                score = "successful"
            elif "unsuccessful" in response_text.lower():
                score = "unsuccessful"
                
            return {
                "type": "evaluation",
                "feedback": response_text.strip(),
                "score": score
            }
        
        # If it contains Python code, it might be an implementation
        if "import pandas" in response_text or "def " in response_text or "import numpy" in response_text:
            # Likely code without the proper format
            print("Detected code-like content")
            return {
                "type": "implementation",
                "code": clean_code(response_text)
            }
        
        return {"type": "error", "message": "Could not determine response format"}

# Main function to run the loop
async def main():
    # Fetch Bitcoin data
    print("Fetching Bitcoin historical data...")
    btc_data = fetch_bitcoin_data(days=365)
    
    if btc_data is None:
        print("Failed to fetch data. Exiting.")
        return
        
    print(f"Fetched data: {len(btc_data)} days of Bitcoin prices")
    
    # Initial strategy generation
    print("Generating trading strategy...")
    strategy_result = await generate_strategy(current_query)
    
    if strategy_result["type"] == "implementation":
        print("\n=== Generated Strategy ===")
        print(strategy_result["code"])
        
        # Backtest the strategy
        print("\n=== Backtesting Strategy ===")
        backtest_results = backtest_strategy(strategy_result["code"], btc_data)
        
        if backtest_results is not None:
            print("\n=== Backtest Results ===")
            print(backtest_results.tail())
            
            # Plot the results if the backtest includes cumulative returns
            if "cumulative_returns" in backtest_results.columns:
                plt.figure(figsize=(12, 6))
                backtest_results["cumulative_returns"].plot()
                plt.title("Strategy Cumulative Returns")
                plt.xlabel("Date")
                plt.ylabel("Return")
                plt.savefig("strategy_returns.png")
                plt.close()
                print("Saved performance chart to strategy_returns.png")
        
        # Evaluate the strategy with backtest results
        print("\n=== Evaluating Strategy ===")
        evaluation_result = await evaluate_strategy(
            strategy_result["code"], 
            backtest_results
        )
        
        if evaluation_result["type"] == "evaluation":
            print(f"Score: {evaluation_result['score']}")
            print(f"Feedback: {evaluation_result['feedback']}")
            
            # If strategy needs refinement, improve it
            if evaluation_result["score"] in ["needs_refinement", "unsuccessful"]:
                print("\n=== Refining Strategy ===")
                
                refinement_query = f"""Improve the following trading strategy based on this feedback: 
                {evaluation_result['feedback']}
                
                Original strategy:
                {strategy_result['code']}
                
                Backtest results:
                {backtest_results.tail().to_string() if backtest_results is not None else "No backtest results available"}
                """
                
                improved_strategy = await generate_strategy(refinement_query)
                
                if improved_strategy["type"] == "implementation":
                    print("\n=== Improved Strategy ===")
                    print(improved_strategy["code"])
                    
                    # Backtest the improved strategy
                    print("\n=== Backtesting Improved Strategy ===")
                    improved_results = backtest_strategy(improved_strategy["code"], btc_data)
                    
                    if improved_results is not None:
                        print("\n=== Improved Backtest Results ===")
                        print(improved_results.tail())
    else:
        print("Error in strategy generation:", strategy_result.get("message", "Unknown error"))

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())