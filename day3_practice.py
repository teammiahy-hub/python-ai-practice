import time
import requests
import json
# 1. Function: safe_api_call(url)
#    - Try to make a request
#    - If it fails, print the error
#    - Try 3 times before giving up
#    - Return None if all fail
def safe_api_call(url,max_retries=3):
    for retries in range(max_retries):
        try:
            # make the request
            response = requests.get(url, timeout=5)
            if response.status_code == 404:
                if retries < 3:
                    raise Exception(f"Temporarily unavailable to visit url {url}")
            return f"{url} valid"
        except Exception as e:
            waiting_time = retries ** 2
            print(f"Waiting time for next retry {waiting_time} sec")
            time.sleep(waiting_time)
    return None

print(safe_api_call("https://github.com"))

# 2. Function: save_results(data, filename)
#    - Save any data as JSON
#    - Print success message
#    - Handle errors if can't save
def save_results(data,filename="data_base.json"):
    try:
        with open(filename,"w") as file:
            json.dump(data,file)
    except FileNotFoundError:
        # optional for this test
        print("File does not exist")

data_content = [
    {"role":"owner","content":"database 1"},
    {"role":"subowner","content":"database 2"}
]

save_results(data_content,"data_base.json")

# 3. Function: load_results(filename)
#    - Load JSON from file
#    - Return empty list if file missing
#    - Return empty list if file is corrupted
def load_results(filename):
    try:
        with open(filename,"r",encoding="utf-8") as file:
            print("The file is valid")
            result = json.load(file)
        return result
    except FileNotFoundError:
        print("File does not exist")
        return []
    except json.JSONDecodeError:
        print("File is not in appropriate format")
        return []

content = load_results("data_base.json")
print(f"The content is {content}")