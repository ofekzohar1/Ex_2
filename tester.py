import subprocess
import time


def parse(x):
    "".join(x.split())  # Remove Whitespaces.


SEPARATOR = "-" * 7
PYTHON_IMPL = "python3 kmeans_pp.py"
tests = ["3 333 test_data/input_1_db_1.txt test_data/input_1_db_2.txt",
         "7 test_data/input_2_db_1.txt test_data/input_2_db_2.txt",
         "15 750 test_data/input_3_db_1.txt test_data/input_3_db_2.txt"]
outputs = ["test_data/output_1.txt", "test_data/output_2.txt", "test_data/output_3.txt"]
for output, test in zip(outputs, tests):
    python_start = time.time()
    python_output = subprocess.check_output(f"{PYTHON_IMPL} {test}", shell=True).decode()
    python_end = time.time()

    with open(output, "r") as f:
        real_output = f.read()

    if parse(python_output) != parse(real_output):
        print(f"Python Output Differs from Real Output!")
        print(f"{SEPARATOR}PYTHON{SEPARATOR}\n{python_output}\n{SEPARATOR}{len('PYTHON') * '-'}{SEPARATOR}")
        print(f"{SEPARATOR}REAL{SEPARATOR}\n{python_output}\n{SEPARATOR}{len('REAL') * '-'}{SEPARATOR}")

    print(f"Python Runtime was: {python_end - python_start}")
