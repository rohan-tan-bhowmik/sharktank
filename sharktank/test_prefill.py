import subprocess
import numpy as np
import torch

def run_python_file(python_file):
    try:
        result = subprocess.run(['python', python_file], check=True)
        print(f"Python file '{python_file}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing '{python_file}': {e}")

def run_bash_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"Bash command '{command}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing bash command '{command}': {e}")

def compare_npy_files(file1, file2):
    try:
        array1 = np.load(file1)
        array2 = np.load(file2)

        print("RMSE: ", np.sqrt(np.mean((array1 - array2) ** 2)))

        torch.testing.assert_close(array1, array2)

        if np.array_equal(array1, array2):
            print(f"Files '{file1}' and '{file2}' are identical.")
        else:
            print(f"Files '{file1}' and '{file2}' differ.")
    except Exception as e:
        print(f"Error comparing files '{file1}' and '{file2}': {e}")

if __name__ == "__main__":
    python_file = 'create_test_prefill.py'
    bash_command_1 = '../../iree/build/tools/iree-compile paged_cached_attention_causal.mlir --iree-hal-target-backends=rocm --iree-hip-target=gfx1100 -o paged_attn.vmfb'
    bash_command_2 = '../../iree/build/tools/iree-run-module --module=paged_attn.vmfb --device=hip --function=prefill_bs4 --input=@attn_q.npy --input=@attn_k.npy --input=@attn_v.npy --input=@arg3.npy --input=@arg4.npy --input=@arg5.npy --output=@attn_out.npy'
    npy_file_1 = 'attn_out.npy'
    npy_file_2 = 'attn_ref.npy'

    run_python_file(python_file)
    run_bash_command(bash_command_1)
    run_bash_command(bash_command_2)
    compare_npy_files(npy_file_1, npy_file_2)