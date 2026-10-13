"""Share exact committed decisions across processes and noise points."""
import json
from pathlib import Path
import sqlite3

import numpy as np

LABELS=("original","strong","single_fault_lookup")


class DecisionCache:
    def __init__(self,path,identity,nd,nl):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.connection=sqlite3.connect(path,timeout=30)
        self.nd,self.nl=nd,nl
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, identity TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS decisions (record BLOB PRIMARY KEY, prediction BLOB NOT NULL)")
        value=json.dumps(dict(identity,nd=nd,nl=nl,labels=LABELS),sort_keys=True)
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES (1,?)",(value,))
            assert self.connection.execute("SELECT identity FROM metadata WHERE id=1").fetchone()[0]==value

    def key(self,record):
        assert record.shape==(self.nd,) and np.all((record==0)|(record==1))
        return np.packbits(record,bitorder="little").tobytes()

    def store(self,records,arrays):
        assert set(arrays)==set(LABELS)
        assert all(a.shape==(len(records),self.nl) for a in arrays.values())
        assert all(np.all((a==0)|(a==1)) for a in arrays.values())
        unique={}
        for i,record in enumerate(records):
            key=self.key(record)
            value=b"".join(np.asarray(arrays[label][i],dtype=np.uint8).tobytes()for label in LABELS)
            if key in unique: assert unique[key]==value,"Conflicting duplicate detector decisions"
            unique[key]=value
        with self.connection:
            self.connection.executemany("INSERT OR IGNORE INTO decisions VALUES (?,?)",unique.items())
            for key,value in unique.items():
                assert self.connection.execute("SELECT prediction FROM decisions WHERE record=?",(key,)).fetchone()[0]==value,"Conflicting committed decision"

    def predictions(self,decoder,detectors):
        records,inverse=np.unique(np.asarray(detectors,dtype=np.uint8),axis=0,return_inverse=True)
        arrays={label:np.zeros((len(records),self.nl),dtype=np.uint8) for label in LABELS}
        missing=[]
        for i,record in enumerate(records):
            row=self.connection.execute("SELECT prediction FROM decisions WHERE record=?",(self.key(record),)).fetchone()
            if row is None: missing.append(i);continue
            value=np.frombuffer(row[0],dtype=np.uint8).reshape(len(LABELS),self.nl)
            for j,label in enumerate(LABELS):arrays[label][i]=value[j]
        if missing:
            computed=decoder.predictions(records[missing])
            self.store(records[missing],computed)
            for label in LABELS:arrays[label][missing]=computed[label]
        return {label:array[inverse]for label,array in arrays.items()}

    def check(self):
        assert self.connection.execute("PRAGMA quick_check").fetchone()==("ok",)

    def close(self):
        self.connection.close()
