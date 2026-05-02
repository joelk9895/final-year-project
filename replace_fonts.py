import re

with open('final/final/ContentView.swift', 'r') as f:
    content = f.read()

# Replace serif fonts
content = re.sub(r'\.font\(\.system\(size:\s*(\d+),\s*weight:\s*\.\w+,\s*design:\s*\.serif\)\)', r'.font(.custom("InstrumentSerif-Regular", size: \1))', content)

# Replace sans fonts with weight
content = re.sub(r'\.font\(\.system\(size:\s*(\d+),\s*weight:\s*(\.\w+)\)\)', r'.font(.custom("InstrumentSans-Regular", size: \1)).fontWeight(\2)', content)

# Replace sans fonts without weight
content = re.sub(r'\.font\(\.system\(size:\s*(\d+)\)\)', r'.font(.custom("InstrumentSans-Regular", size: \1))', content)

with open('final/final/ContentView.swift', 'w') as f:
    f.write(content)
