"""Retrieve primary sources and preserve their exact text."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import re
import urllib.request
from .artifacts import WORK, commit_json, digest


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.urls=[]
    def handle_starttag(self, tag, attributes):
        if tag == "a":
            self.urls.extend(value for key,value in attributes if key == "href" and value)


def retrieve(identifier):
    url = "https://arxiv.org/html/"+identifier+"v1"
    try:
        request = urllib.request.Request(url, headers={"User-Agent":"ISCAS-research-source-verification/1.0"})
        raw = urllib.request.urlopen(request,timeout=45).read()
        folder = WORK / "provenance/literature"
        folder.mkdir(parents=True,exist_ok=True)
        path = folder / (identifier+"v1.html")
        path.write_bytes(raw)
        parser = Links(); parser.feed(raw.decode("utf-8",errors="replace"))
        return dict(identifier=identifier,url=url,sha256=digest(path),
                    code_urls=[u for u in parser.urls if "github" in u or "zenodo" in u],
                    path=str(path))
    except Exception as error:
        return dict(identifier=identifier,url=url,error=str(error))


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(retrieve,["2603.01628","2509.00850"]))
    commit_json(WORK/"provenance/literature/primary-sources.json",records)
    print(records,flush=True)
