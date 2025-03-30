// API endpoint base URL
const API_BASE_URL = 'http://localhost:5000';

// Current strategy code and backtest results
let currentStrategyCode = '';
let backtestResults = null;

// DOM Elements
document.addEventListener('DOMContentLoaded', function() {
  // Tab navigation
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      // Deactivate all tabs
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      // Activate selected tab
      tab.classList.add('active');
      const tabId = tab.getAttribute('data-tab');
      document.getElementById(`${tabId}-tab`).classList.add('active');
    });
  });
  
  // Generate Strategy Button
  document.getElementById('generate-btn').addEventListener('click', generateStrategy);
  
  // Backtest Button
  document.getElementById('backtest-btn').addEventListener('click', backtestStrategy);
  
  // Evaluate Button
  document.getElementById('evaluate-btn').addEventListener('click', evaluateStrategy);
  
  // Logs Buttons
  document.getElementById('fetch-logs-btn').addEventListener('click', fetchLogs);
  document.getElementById('copy-logs-btn').addEventListener('click', copyLogs);
  
  // Toggle Debug Button
  document.getElementById('toggle-debug-btn').addEventListener('click', function() {
    const debugOutput = document.getElementById('debug-output');
    debugOutput.style.display = debugOutput.style.display === 'none' ? 'block' : 'none';
  });
});

// Generate a trading strategy
async function generateStrategy() {
  const query = document.getElementById('strategy-query').value;
  if (!query) {
    alert('Please enter a strategy description');
    return;
  }
  
  try {
    // Show loading state
    document.getElementById('generate-btn').textContent = 'Generating...';
    document.getElementById('generate-btn').disabled = true;
    
    const response = await fetch(`${API_BASE_URL}/generate-strategy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query })
    });
    
    const result = await response.json();
    
    if (result.type === 'implementation') {
      currentStrategyCode = result.code;
      document.getElementById('strategy-code').textContent = result.code;
      document.getElementById('strategy-result').style.display = 'block';
      
      // Switch to backtest tab
      document.querySelectorAll('.tab')[1].click();
    } else {
      alert('Error: ' + (result.message || 'Failed to generate strategy'));
    }
  } catch (error) {
    console.error('Error:', error);
    alert('Error: ' + error.message);
  } finally {
    document.getElementById('generate-btn').textContent = 'Generate Strategy';
    document.getElementById('generate-btn').disabled = false;
  }
}

// Backtest a trading strategy
async function backtestStrategy() {
  if (!currentStrategyCode) {
    alert('No strategy code available. Please generate a strategy first.');
    return;
  }
  
  try {
    // Show loading state
    document.getElementById('backtest-btn').textContent = 'Running...';
    document.getElementById('backtest-btn').disabled = true;
    
    const days = document.getElementById('backtest-days').value;
    
    const response = await fetch(`${API_BASE_URL}/backtest-strategy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ 
        code: currentStrategyCode,
        days: parseInt(days)
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      backtestResults = result.results;
      
      // Display last 5 results
      const lastFiveResults = backtestResults.slice(-5);
      document.getElementById('backtest-output').textContent = 
        JSON.stringify(lastFiveResults, null, 2);
      document.getElementById('backtest-result').style.display = 'block';
    } else {
      alert('Backtest failed: ' + (result.message || 'No results generated'));
      document.getElementById('backtest-output').textContent = 
        'Backtest did not produce results. Check server logs for details.';
      document.getElementById('backtest-result').style.display = 'block';
    }
  } catch (error) {
    console.error('Error:', error);
    alert('Error: ' + error.message);
  } finally {
    document.getElementById('backtest-btn').textContent = 'Run Backtest';
    document.getElementById('backtest-btn').disabled = false;
  }
}

// Evaluate a trading strategy with fallback
async function evaluateStrategy() {
  if (!currentStrategyCode) {
    alert('No strategy code available. Please generate a strategy first.');
    return;
  }
  
  try {
    // Show loading state
    document.getElementById('evaluate-btn').textContent = 'Evaluating...';
    document.getElementById('evaluate-btn').disabled = true;
    
    // First try the regular evaluation endpoint
    let response;
    let result;
    let usedFallback = false;
    
    try {
      response = await fetch(`${API_BASE_URL}/evaluate-strategy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          code: currentStrategyCode,
          backtest_results: backtestResults
        })
      });
      
      result = await response.json();
      console.log("Regular evaluation result:", result);
      
      // Check if we got a valid response
      if (!result.type || result.type !== 'evaluation' || result.error) {
        throw new Error("Invalid response format");
      }
    } catch (regularError) {
      console.error("Regular evaluation failed:", regularError);
      
      // Use the fallback evaluation
      console.log("Trying fallback evaluation...");
      usedFallback = true;
      
      response = await fetch(`${API_BASE_URL}/simple-evaluate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ code: currentStrategyCode })
      });
      
      result = await response.json();
      console.log("Fallback evaluation result:", result);
    }
    
    // Display the results
    if (result.type === 'evaluation') {
      document.getElementById('eval-score').textContent = result.score || 'N/A';
      document.getElementById('eval-feedback').textContent = 
        (usedFallback ? "⚠️ Using fallback evaluation: " : "") + 
        (result.feedback || 'No feedback provided');
      document.getElementById('evaluation-result').style.display = 'block';
    } else {
      alert('Evaluation failed: Unable to get valid evaluation');
    }
  } catch (error) {
    console.error('Network Error:', error);
    alert('Network error: ' + error.message);
  } finally {
    document.getElementById('evaluate-btn').textContent = 'Evaluate Strategy';
    document.getElementById('evaluate-btn').disabled = false;
  }
}

// Fetch LLM logs
async function fetchLogs() {
  try {
    const response = await fetch(`${API_BASE_URL}/get-logs`);
    const logs = await response.json();
    
    document.getElementById('llm-logs').textContent = 
      JSON.stringify(logs, null, 2);
  } catch (error) {
    console.error('Error:', error);
    document.getElementById('llm-logs').textContent = 
      'Error fetching logs: ' + error.message;
  }
}

// Copy logs to clipboard
function copyLogs() {
  const logText = document.getElementById('llm-logs').textContent;
  navigator.clipboard.writeText(logText)
    .then(() => {
      alert('Logs copied to clipboard!');
    })
    .catch(err => {
      console.error('Failed to copy logs:', err);
      alert('Failed to copy logs: ' + err.message);
    });
}

// Function to log debug info
function logDebug(message, data = null) {
  console.log(message, data);
  const debugPanel = document.getElementById('debug-panel');
  const debugOutput = document.getElementById('debug-output');
  
  let debugText = debugOutput.textContent;
  debugText += `\n[${new Date().toISOString()}] ${message}`;
  if (data) {
    debugText += '\n' + JSON.stringify(data, null, 2);
  }
  debugText += '\n------------------';
  
  debugOutput.textContent = debugText;
  debugPanel.style.display = 'block';
} 