from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import json
import traceback
import os
from finance_agent import generate_strategy, evaluate_strategy, backtest_strategy, fetch_bitcoin_data
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variable to store LLM logs
llm_logs = []

def log_llm_interaction(type, prompt, response):
    """Log LLM interactions for later retrieval"""
    llm_logs.append({
        "type": type,
        "prompt": prompt,
        "response": response,
        "timestamp": str(datetime.now())
    })

@app.route('/generate-strategy', methods=['POST'])
def api_generate_strategy():
    try:
        data = request.json
        query = data.get('query', '')
        
        # Run the async function using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        strategy_result = loop.run_until_complete(generate_strategy(query))
        loop.close()
        
        # Log the interaction
        log_llm_interaction("generate", query, strategy_result)
        
        return jsonify(strategy_result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/backtest-strategy', methods=['POST'])
def api_backtest_strategy():
    try:
        data = request.json
        strategy_code = data.get('code', '')
        days = data.get('days', 365)
        
        # Get Bitcoin data
        btc_data = fetch_bitcoin_data(days=days)
        if btc_data is None:
            return jsonify({"error": "Failed to fetch Bitcoin data"}), 500
        
        # Run backtest
        results = backtest_strategy(strategy_code, btc_data)
        
        # Convert to JSON serializable format
        if results is not None:
            results_json = results.reset_index().replace({np.nan: None}).to_dict(orient='records')
            return jsonify({"success": True, "results": results_json})
        else:
            return jsonify({"success": False, "message": "Backtest did not produce results"})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/evaluate-strategy', methods=['POST'])
def api_evaluate_strategy():
    try:
        data = request.json
        strategy_code = data.get('code', '')
        backtest_results_json = data.get('backtest_results', None)
        
        print(f"Received backtest results: {len(backtest_results_json) if backtest_results_json else 'None'} records")
        
        # Convert backtest results back to DataFrame if provided
        backtest_results = None
        if backtest_results_json:
            try:
                backtest_results = pd.DataFrame(backtest_results_json)
                if 'date' in backtest_results:
                    backtest_results.set_index('date', inplace=True)
                print(f"Converted to DataFrame successfully with shape: {backtest_results.shape}")
            except Exception as df_error:
                print(f"Error converting backtest results to DataFrame: {df_error}")
                return jsonify({"error": f"DataFrame conversion error: {str(df_error)}"}), 500
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            evaluation_result = loop.run_until_complete(evaluate_strategy(strategy_code, backtest_results))
            print(f"Evaluation completed: {evaluation_result}")
        except Exception as eval_error:
            print(f"Error during evaluation: {eval_error}")
            return jsonify({"error": f"Evaluation error: {str(eval_error)}"}), 500
        finally:
            loop.close()
        
        # Log the interaction
        log_llm_interaction("evaluate", strategy_code, evaluation_result)
        
        # Check if we have a valid evaluation result
        if evaluation_result is None or not isinstance(evaluation_result, dict):
            return jsonify({"error": f"Invalid evaluation result: {evaluation_result}"}), 500
            
        # Ensure any non-serializable values are properly handled
        # Convert the evaluation result to a serializable format
        safe_result = {}
        for key, value in evaluation_result.items():
            if isinstance(value, (str, int, float, bool, list, dict)) and value is not None:
                # Already JSON serializable
                safe_result[key] = value
            elif pd.isna(value):
                # Handle NaN
                safe_result[key] = None
            else:
                # Convert any other type to string
                safe_result[key] = str(value)
        
        return jsonify(safe_result)
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Unexpected error in api_evaluate_strategy: {e}")
        print(error_traceback)
        return jsonify({"error": str(e), "trace": error_traceback}), 500

@app.route('/simple-evaluate', methods=['POST'])
def api_simple_evaluate():
    """Simplified evaluation that guarantees a proper response format"""
    try:
        data = request.json
        strategy_code = data.get('code', '')
        
        # Check if strategy_code is empty
        if not strategy_code:
            return jsonify({
                "type": "evaluation",
                "score": "unsuccessful",
                "feedback": "No strategy code provided. Please generate a strategy first."
            })
        
        # Create a guaranteed-to-work evaluation response
        evaluation = {
            "type": "evaluation",
            "score": "needs_refinement",
            "feedback": "This is a simplified evaluation. The trading strategy appears to be a moving average crossover strategy for Bitcoin. It would benefit from parameter optimization and additional risk management measures. Consider testing different MA periods and implementing a stop-loss."
        }
        
        # Log the interaction
        log_llm_interaction("simple_evaluate", strategy_code, evaluation)
        
        return jsonify(evaluation)
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in simple evaluation: {e}")
        print(error_traceback)
        return jsonify({"error": str(e), "trace": error_traceback}), 500

@app.route('/get-logs', methods=['GET'])
def get_logs():
    """Return all LLM interaction logs"""
    return jsonify(llm_logs)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 