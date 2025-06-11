from tabulate import tabulate


def profile_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper


def table_results(results):
    headers = ["Size (floats)", "Size (KB)", "Avg Time / call", "Avg Time on all ranks"]
    print(tabulate(results, headers=headers, tablefmt="pipe"))