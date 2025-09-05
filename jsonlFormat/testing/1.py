import requests, gzip
import pandas as pd
from io import BytesIO


url = "https://www.flipkart.com/sitemap_v_view-browse.xml.gz"
# url = "https://www.google.com/sitemap.xml"
# url = "https://www.amazon.com/sitemap.xml"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

try:
    res = requests.get(url, headers=headers, timeout=15)
    res.raise_for_status()

    content = res.content
    # Check if gzipped
    if url.endswith(".gz") or res.headers.get("Content-Encoding") == "gzip":
        xml_content = gzip.decompress(content).decode("utf-8", errors="ignore")
    else:
        xml_content = content.decode("utf-8", errors="ignore")

    # Parse XML into DataFrame
    df = pd.read_xml(BytesIO(xml_content.encode("utf-8")))

    if "loc" in df.columns:
        # Convert the 'loc' column to a list and write to a text file
        urls = df['loc'].tolist()
        with open('urls.txt', 'w') as f:
            for url in urls:
                f.write(url + '\n')
        print("Total URLs found:", len(df))
        print(df["loc"].head(10))  # first 10 URLs
    else:
        print("'loc' column not found in sitemap XML")

except Exception as e:
    print(" Error:", e)
