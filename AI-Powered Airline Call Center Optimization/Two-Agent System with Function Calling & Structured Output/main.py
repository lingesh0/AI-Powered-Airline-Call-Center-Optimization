import json
import re

def get_flight_info(flight_number: str) -> dict:
    """Simulates fetching flight data."""
    flight_db = {
        "AI123": {"flight_number": "AI123", "departure_time": "08:00 AM", "destination": "Delhi", "status": "Delayed"},
        "BA456": {"flight_number": "BA456", "departure_time": "10:30 AM", "destination": "London", "status": "On Time"},
        "UA789": {"flight_number": "UA789", "departure_time": "02:15 PM", "destination": "New York", "status": "Cancelled"}
    }
    return flight_db.get(flight_number, None)

def info_agent_request(flight_number: str) -> str:
    """Fetches flight data and returns JSON response."""
    flight_info = get_flight_info(flight_number)
    if flight_info:
        return json.dumps(flight_info)
    else:
        return json.dumps({"error": "Flight not found"})

def extract_flight_number(query: str) -> str:
    """Extracts flight number from user query."""
    match = re.search(r'Flight\s([A-Z]{2}\d{3})', query)
    return match.group(1) if match else ""

def qa_agent_respond(user_query: str) -> str:
    """Processes user query and returns structured JSON response."""
    flight_number = extract_flight_number(user_query)
    if not flight_number:
        return json.dumps({"answer": "No flight number found in query."})
    
    flight_data_json = info_agent_request(flight_number)
    flight_data = json.loads(flight_data_json)
    
    if "error" in flight_data:
        return json.dumps({"answer": f"Flight {flight_number} not found in database."})
    
    response = {
        "answer": f"Flight {flight_data['flight_number']} departs at {flight_data['departure_time']} to {flight_data['destination']}. Current status: {flight_data['status']}."
    }
    return json.dumps(response)

# Test Cases
if __name__ == "__main__":
    print(get_flight_info("AI123"))  # Expected: Flight details dictionary
    print(info_agent_request("AI123"))  # Expected: JSON string
    print(qa_agent_respond("When does Flight AI123 depart?"))  # Expected JSON response
    print(qa_agent_respond("What is the status of Flight AI999?"))  # Expected JSON response (Flight not found)