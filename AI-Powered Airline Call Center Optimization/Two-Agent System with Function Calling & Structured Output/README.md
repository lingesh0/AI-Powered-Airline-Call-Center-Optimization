# Problem 1: Two-Agent System with Function Calling & Structured Output

## **Overview**
This project implements a two-agent AI system for answering user queries about airline flights.  
It consists of:
1. **Info Agent** – Fetches flight details in JSON format.
2. **QA Agent** – Processes user queries and returns structured responses.



2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows


3. Install Dependencies

pip install -r requirements.txt


Usage
Run the main program using:

python main.py


Configuration

Ensure all required libraries are installed from requirements.txt.



Example Queries
"What time does Flight AI123 depart?"

"What is the status of Flight AI999?"

The system will return structured JSON responses based on available flight data.