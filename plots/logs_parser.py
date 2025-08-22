import os 
import numpy as np
log_folder = "/home/vtordjma/Documents/log_expe_icra"
folders = os.listdir(log_folder)
folders = [x for x in folders if "CAM" not in x]
folders = [x for x in folders if "output" not in x]
data= []
methods = ["foa", "rl"]
import re
for folder in folders : 
    folder_path = os.path.join(log_folder, folder)
    files = os.listdir(folder_path)
    for file in files : 
        actions= []
        with open(os.path.join(folder_path, file), "r") as f:
            log = f.read()
        lines  = log.split("\n")
        lines = [x for x in lines if "Joystick=[ 0.001 -1.   ]" in x]
        for line in lines:
            match = re.search(r"Action=\[(.*?)\], Joystick", line)
            if match:
                action_str = match.group(1).strip()
                action = [float(v) for v in action_str.split()]
                actions.append(action)
        method = "foa" if "foa" in file else "rl"
        timestamp = file.split(method)[-1][1:]
        data.append({'experiment': folder, "method":method,"timestamp":timestamp, 'mean_action': np.mean(abs(np.array(actions)))})

import pandas as pd
df = pd.DataFrame(data)
df = df.sort_values(by=["method", "timestamp"]) 
df.to_csv(os.path.join(log_folder, "output/actions.csv"), index=False)
print("Data saved to actions.csv")