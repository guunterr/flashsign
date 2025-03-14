import modal # type: ignore
import subprocess
import os
image = modal.Image.from_registry(f'nvidia/cuda:12.4.0-devel-ubuntu22.04', add_python="3.12")
app = modal.App(name="profile", image=image)
    
@app.local_entrypoint()
def main():
    tempcu = run_command("cat temp.cu", capture_output=True, text=True).stdout
    utilscu = run_command("cat utils.cu", capture_output=True, text=True).stdout
    doprofile = run_command("cat do_profile.sh", capture_output=True, text=True).stdout
    # check_nvidia_smi.remote()
    profile_temp.remote([tempcu, utilscu, doprofile])
    # sb = modal.Sandbox.create(app=app, image=image)
    # copy_file(sb, 'temp.cu')
    # copy_file(sb, 'utils.cu')
    # run_command(sb, "nvcc ~/temp.cu -o ~/temp").wait()
    # print(run_command(sb, "which nvcc").stderr.read())
    # print(run_command(sb, "ls ~").stdout.read())
    # sb.terminate()
    
@app.function(gpu='T4')
def profile_temp(files):
    print(os.getcwd())
    print(run_command("which bash", capture_output=True, text=True).stdout)
    print(run_command("ls", capture_output=True, text=True).stdout)
    copy_file(files[0], 'temp.cu')
    copy_file(files[1], 'utils.cu')
    copy_file(files[2], 'do_profile.sh')
    run_command("chmod +x do_profile.sh")
    print(run_command("which sudo"))
    output = run_command("./do_profile.sh", capture_output=True)
    print(output.stdout)
    print(output.stderr)
    
@app.function(gpu='any')
def check_nvidia_smi():
    try:
        print("Checking nvidia-smi")
        output = subprocess.run(["nvidia-smi"], text=True, capture_output=True)
        print(output)
    except Exception as e:
        print(e)
    
def run_command(cmd, text=True, capture_output = True,**kwargs ):
    print(cmd)
    return subprocess.run(cmd.split(" "), text=text, capture_output=capture_output, **kwargs)
        
def copy_file(code, dest):
    with open(dest, 'w') as file:
        file.write(str(code))
        
if __name__ == "__main__":
    
    pass