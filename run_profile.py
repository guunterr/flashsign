import modal # type: ignore
import subprocess
import os
image = modal.Image.from_registry(f'nvidia/cuda:12.4.0-devel-ubuntu22.04', add_python="3.12")
app = modal.App(name="profile", image=image)
    
@app.local_entrypoint()
def main():
    files_to_write = ['temp.cu', 'utils.cu', 'do_profile.sh', 
                      'kernels/kernel1.cuh', 'kernels/kernel2.cuh', 'kernels/kernel3.cuh',
                      'kernels/kernel4.cuh', 'kernels/kernel5.cuh']
    file_contents = [(run_command(f"cat {file}", capture_output=True, text=True).stdout, file) for file in files_to_write]
    profile_temp.remote(file_contents)
    # check_nvidia_smi.remote()
    # sb = modal.Sandbox.create(app=app, image=image)
    # copy_file(sb, 'temp.cu')
    # copy_file(sb, 'utils.cu')
    # run_command(sb, "nvcc ~/temp.cu -o ~/temp").wait()
    # print(run_command(sb, "which nvcc").stderr.read())
    # print(run_command(sb, "ls ~").stdout.read())
    # sb.terminate()
    
@app.function(gpu='T4')
def profile_temp(files):
    run_command("mkdir kernels")
    for file in files:
        copy_file(file[0], file[1])
    print(run_command("ls", capture_output=True, text=True).stdout)
    print(run_command("ls kernels", capture_output=True, text=True).stdout)
    run_command("chmod +x do_profile.sh")
    output = run_command("./do_profile.sh", capture_output=True)
    print(output.stdout)
    print(output.stderr)
    
def run_command(cmd, text=True, capture_output = True,**kwargs ):
    print(cmd)
    return subprocess.run(cmd.split(" "), text=text, capture_output=capture_output, **kwargs)
        
def copy_file(code, dest):
    with open(dest, 'w') as file:
        file.write(str(code))
        
if __name__ == "__main__":
    
    pass