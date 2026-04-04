from stock import run_pipeline
import json

result = run_pipeline()

print("Pipeline Output:")
print(result)

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Saved to result.json successfully!")