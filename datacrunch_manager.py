import os
import subprocess
import time
import datacrunch
from datacrunch import DataCrunchClient
import datacrunch.constants

CLIENT_SECRET = os.environ.get('DATACRUNCH_CLIENT_SECRET')
CLIENT_ID = "RFhYGYM8Ag6o4zsmoHb14"
OS_VOLUME_ID = "bdd1921d-1c55-4fdb-917e-1373a310253d"

dc_client = DataCrunchClient(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

ssh_keys = dc_client.ssh_keys.get()
ssh_keys_ids = list(map(lambda ssh_key: ssh_key.id, ssh_keys))

def run_profile(ip):
    process = subprocess.run([f"./run_datacrunch.sh", f"{ip}"], capture_output=True, text=True, shell=True)
    print(process.stdout)
    
def create_instance():
    if len(dc_client.instances.get()) == 0:
        print("Creating instance...")
        instance = dc_client.instances.create(
                                                instance_type="1A100.22V",
                                                # instance_type="1H100.80S.30V", 
                                                image=OS_VOLUME_ID,
                                                # ssh_key_ids=ssh_keys_ids,
                                                is_spot=True,
                                                hostname="test-instance",
                                                description="Testing SDK")
        print("Waiting for instance to start...")
        while True:
            if instance.status == datacrunch.constants.InstanceStatus.RUNNING:
                break
            instance = dc_client.instances.get()[0]
            print(f"Instance status: {instance.status}")
            time.sleep(5)
        instance = dc_client.instances.get()[0]
        print(f"Instance {instance.status}, ip = {instance.ip}")
        os.environ["DATACRUNCH_IP"] = instance.ip
        # 
    else:
        instance = dc_client.instances.get()[0]
        print(f"Instance {instance.status}, ip = {instance.ip}")
        os.environ["DATACRUNCH_IP"] = instance.ip
    return instance.ip

def delete_instance():
    if len(dc_client.instances.get()) > 0:
        instance = dc_client.instances.get()[0]
        print(f"Instance {instance.status}, ip = {instance.ip}")
        print(f"Hibernating instance")
        dc_client.instances.action(instance.id, dc_client.constants.instance_actions.HIBERNATE)
    else:
        print(f"No instance")

if __name__ == "__main__":
    user_inpt = input("create (c) or delete (d)?")
    if user_inpt.lower() == "c":
        create_instance()
    else:
        delete_instance()