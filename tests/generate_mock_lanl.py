import gzip
import os

data_dir = "/data"
auth_path = os.path.join(data_dir, "auth.txt.gz")
redteam_path = os.path.join(data_dir, "redteam.txt")

redteam_events = [
    (100, "U1@dom", "C1", "C2"),
    (200, "U1@dom", "C2", "C3"),
]

# We need to make sure there's enough events to bypass benign stride and window padding
# auth.txt: time, srcUser@dom, dstUser@dom, srcComp, dstComp, authType, logonType, authOrient, success
with open(redteam_path, "wt") as f:
    for t, u, s, d in redteam_events:
        f.write(f"{t},{u},{s},{d}\n")

with gzip.open(auth_path, "wt") as f:
    # A few benign events outside the redteam window so that window logic is happy
    f.write(f"10,U2@dom,U2@dom,C4,C4,NTLM,Network,LogOn,Success\n")
    # Redteam
    for t, u, s, d in redteam_events:
        f.write(f"{t},{u},?,{s},{d},Kerberos,Network,LogOn,Success\n")
        # Benign padding to make benign_stride work
        for i in range(20):
            f.write(f"{t+i+1},U2@dom,U2@dom,C4,C4,NTLM,Network,LogOn,Success\n")
            
    f.write(f"300,U2@dom,U2@dom,C4,C4,NTLM,Network,LogOn,Success\n")
        
print("Mock LANL dataset generated in /data")
