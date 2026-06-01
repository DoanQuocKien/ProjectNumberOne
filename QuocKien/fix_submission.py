import pickle
import json

# 1. Load your current dictionary
with open("D:\\CS116\\ProjectNumberOne\\QuocKien\\submission_int_keys_fixed.pkl", "rb") as f:
    submission = pickle.load(f)

clean_submission = {}

for uid, items in submission.items():
    # A. Fix any accidental nested lists (e.g. [['item1', 'item2']] -> ['item1', 'item2'])
    if isinstance(items, list) and len(items) > 0 and isinstance(items[0], list):
        items = [x for sublist in items for x in sublist]
    
    # B. Ensure all items inside are definitely strings
    clean_items = [str(x) for x in items]
    
    # C. APPLY THE TUPLE HACK: Convert the list to a tuple to make it hashable!
    # If the grader's pandas script is crashing, this prevents the crash.
    clean_submission[uid] = tuple(clean_items) 

# 2. Save it back as a pickle (if PKL is strictly required)
with open("submission_bulletproof.pkl", "wb") as f:
    pickle.dump(clean_submission, f)

# 3. HIGHLY RECOMMENDED: Save a JSON version just in case! 
# The prompt explicitly mentioned JSON, and JSON natively handles lists perfectly.
with open("submission.json", "w") as f:
    json.dump(clean_submission, f)

print("Sanitization complete! Try submitting 'submission_bulletproof.pkl' first.")
print("If it still fails, submit 'submission.json'.")