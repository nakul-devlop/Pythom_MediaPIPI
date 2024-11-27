import threading
from flask import Flask, render_template, jsonify
import subprocess
import os
import sys

app = Flask(__name__)

# Store the status of the task globally
task_status = "Not started"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_script')
def run_script():
    # Function to run the demo script
    def run_demo_script():
        global task_status
        script_path = os.path.join(os.path.dirname(__file__), 'ringv5.py')  # Path to the script
        
        # Print the script path for debugging purposes
        print(f"Script path: {script_path}")

        try:
            # Full path to Python executable, you can use sys.executable or specify explicitly
            # python_path = sys.executable  # This should work in most cases; alternatively, you can specify the full path like below
            python_path = "C:\\Users\\Nakul\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"  # Use the correct path to Python
            python_path = "\\ys1cs0bwomgt\\public_html\\mandarjainsanghdelhi.in\\virtualenv\\filas_folder\\3.11\\bin\\python.exe"  # Use the correct path to Python

            result = subprocess.run([python_path, script_path], capture_output=True, text=True)
            
            # Check the result of the script execution
            if result.returncode == 0:
                task_status = "Script ran successfully!"
            else:
                task_status = f"Script failed with error: {result.stderr}"
        except Exception as e:
            task_status = f"An error occurred: {str(e)}"
    
    # Start the function in a separate thread to avoid blocking Flask
    thread = threading.Thread(target=run_demo_script)
    thread.start()

    return "Script is running in the background."

@app.route('/get_status')
def get_status():
    # Return the current status of the background task
    return jsonify(status=task_status)

if __name__ == '__main__':
    app.run(debug=True)
