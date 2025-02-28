import problems
import time
from inspect import getmembers, isfunction
import os

def main():
    functions_list = getmembers(problems, isfunction)
    functions = {}
    for function in functions_list:
        if function[0][:7] == "problem":
            functions[function[0]] = function[1]
    file = "log.txt"
    log = ""
    T = time.process_time()
    for name, function in functions.items():
        t = time.process_time()
        answer = function()
        time_elapsed = time.process_time() - t
        text = f"{name}\n\t Time: {time_elapsed}\n\t Answer: {answer}\n"
        log += text
        print(text)
    total_time_elapsed = time.process_time() - T
    text = f"Total time: {total_time_elapsed}"
    log += text
    print(text)

    # Record results
    f = os.open(file, os.O_WRONLY)
    os.write(f, log.encode())
    os.close(f)


def time_test():
    for problem in [problems.problem_073]:
        t = time.process_time()
        N=10
        solution = problem()
        time_elapsed = time.process_time() - t
        print(N, solution, time_elapsed)

# main()
time_test()

