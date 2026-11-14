"""Publish the common-data fingerprint after the expert collector has finished."""
import json,time,hashlib
from .evaluate import OUT,save

def main():
    while not (OUT/'expert_data_complete.json').exists():time.sleep(2)
    paths=sorted((OUT/'data').glob('route_*.npz'));assert len(paths)==24,len(paths)
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    save(OUT/'dataset_fingerprint.json',dict(sha256=digest,files=files))
    data=json.loads((OUT/'expert_data_complete.json').read_text());data['dataset_sha256']=digest;save(OUT/'expert_data_complete.json',data)
    print('EXPERT_DATA_FINGERPRINT',digest,flush=True)
if __name__=='__main__':main()
