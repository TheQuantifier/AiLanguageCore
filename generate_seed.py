capitals = [
    ('Alabama', 'Montgomery'),
    ('Alaska', 'Juneau'),
    ('Arizona', 'Phoenix'),
    ('Arkansas', 'Little Rock'),
    ('Colorado', 'Denver'),
    ('Connecticut', 'Hartford'),
    ('Delaware', 'Dover'),
    ('Georgia', 'Atlanta'),
    ('Hawaii', 'Honolulu'),
    ('Idaho', 'Boise'),
    ('Illinois', 'Springfield'),
    ('Indiana', 'Indianapolis'),
    ('Iowa', 'Des Moines'),
    ('Kansas', 'Topeka'),
    ('Kentucky', 'Frankfort'),
    ('Louisiana', 'Baton Rouge'),
    ('Maine', 'Augusta'),
    ('Maryland', 'Annapolis'),
    ('Massachusetts', 'Boston'),
    ('Michigan', 'Lansing'),
    ('Minnesota', 'Saint Paul'),
    ('Mississippi', 'Jackson'),
    ('Missouri', 'Jefferson City'),
    ('Montana', 'Helena'),
    ('Nebraska', 'Lincoln'),
    ('Nevada', 'Carson City'),
    ('New Hampshire', 'Concord'),
    ('New Jersey', 'Trenton'),
    ('New Mexico', 'Santa Fe'),
    ('North Carolina', 'Raleigh'),
    ('North Dakota', 'Bismarck'),
    ('Ohio', 'Columbus'),
    ('Oklahoma', 'Oklahoma City'),
    ('Oregon', 'Salem'),
    ('Pennsylvania', 'Harrisburg'),
    ('Rhode Island', 'Providence'),
    ('South Carolina', 'Columbia'),
    ('South Dakota', 'Pierre'),
    ('Tennessee', 'Nashville'),
    ('Utah', 'Salt Lake City'),
    ('Vermont', 'Montpelier'),
    ('Virginia', 'Richmond'),
    ('Washington', 'Olympia'),
    ('West Virginia', 'Charleston'),
    ('Wisconsin', 'Madison'),
    ('Wyoming', 'Cheyenne')
]

math_problems = [
    ("What is 3 + 4?", "3 + 4 equals 7."),
    ("What is 6 times 7?", "6 times 7 equals 42."),
    ("What is 12 divided by 3?", "12 divided by 3 equals 4."),
    ("What is 8 minus 2?", "8 minus 2 equals 6."),
    ("What is 9 squared?", "9 squared equals 81."),
    ("What is the square root of 25?", "The square root of 25 is 5."),
]

countries = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("United Kingdom", "London"),
    ("Canada", "Ottawa"),
    ("Australia", "Canberra"),
    ("Japan", "Tokyo"),
    ("China", "Beijing"),
    ("India", "New Delhi"),
]

print("  {")
print('    "user_input": "What is the capital of Alabama?",')
print('    "response_type": "DIRECT_ANSWER",')
print('    "reason": "This is a factual question about geography that can be answered directly.",')
print('    "response": "The capital of Alabama is Montgomery."')
print("  },")

for state, capital in capitals[1:]:
    print("  {")
    print(f'    "user_input": "What is the capital of {state}?",')
    print('    "response_type": "DIRECT_ANSWER",')
    print('    "reason": "This is a factual question about geography that can be answered directly.",')
    print(f'    "response": "The capital of {state} is {capital}."')
    print("  },")

for question, answer in math_problems:
    print("  {")
    print(f'    "user_input": "{question}",')
    print('    "response_type": "DIRECT_ANSWER",')
    print('    "reason": "This is a basic math problem that can be solved directly.",')
    print(f'    "response": "{answer}"')
    print("  },")

for country, capital in countries:
    print("  {")
    print(f'    "user_input": "What is the capital of {country}?",')
    print('    "response_type": "DIRECT_ANSWER",')
    print('    "reason": "This is a factual question about geography that can be answered directly.",')
    print(f'    "response": "The capital of {country} is {capital}."')
    print("  },")