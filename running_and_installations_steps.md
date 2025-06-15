* Linux Environment: WSL provides a real Linux environment (like Ubuntu) running on top of Windows. This means you can use Linux commands, install Linux packages, and most importantly, install tflite_runtime as if you were on a native Linux machine or a Raspberry Pi.
* Steps to Install tflite_runtime and Run Your Code in WSL:

1- Open your WSL Terminal:
* open Command Prompt or PowerShell and type wsl and press Enter.

2. Navigate to your Project Directory in WSL:
Your Windows drives are typically mounted under /mnt/ in WSL.

* C: drive is /mnt/c
* D: drive is /mnt/d

So, to get to your project directory:
cd /mnt/d/programing/ML_and_DL/deployment_DL/tflite-Rasperry-pi

3. Set up a Python Virtual Environment (Highly Recommended in WSL too):
# Update package lists
sudo apt update

# Install python3-venv if you don't have it (needed for virtual environments)
```
sudo apt install python3-venv
```
# Create a new virtual environment (e.g., named 'wsl_tf_env')
```
python3 -m venv wsl_tf_env
```
# Activate the virtual environment
```
source wsl_tf_env/bin/activate
```
(You should see (wsl_tf_env) appear before your prompt, indicating it's active.)

4. Install tflite_runtime:
```
(wsl_tf_env) pip install tflite_runtime
```
5. Run your Python Script:
Now, execute your classify.py script. Make sure your dog.jpg file is also accessible in that directory.
```
(wsl_tf_env) python3 classify.py --filename dog.jpg --model_path mobilenet_v1_1.0_224_quant.tflite --label_path labels_mobilenet_quant_v1_224.txt
```
