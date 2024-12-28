import problems
import time
from inspect import getmembers, isfunction

def main():
    functions_list = getmembers(problems, isfunction)
    functions = {}
    for function in functions_list:
        if function[0][:7] == "problem":
            functions[function[0]] = function[1]

    for name, function in functions.items():
        time_start = time.time()
        answer = function()
        time_end = time.time()
        time_total = time_end - time_start
        print(f"{name}\n\t Time: {time_total}\n\t Answer: {answer}")

main()

