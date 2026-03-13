import json
fully_correct = 0
format_correct_only = 0
format_wrong = 0

with open("math_baseline.json", "r", encoding = "utf-8") as f:
    data = json.load(f)
    for item in data:
        item = item["scores"]
        if item["format_reward"] == 1.0:
            if item["answer_reward"] == 1.0 :
                fully_correct += 1
            else:
                format_correct_only += 1
        else: 
            format_wrong += 1

print(f"""fully_correct: {fully_correct}; 
      format_correct_only: {format_correct_only}; 
      format_wrong: {format_wrong}""")