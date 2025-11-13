import os
import subprocess
import requests
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

FILE_URLS = {
    "Feynman_with_units.tar.gz": "https://www.dropbox.com/s/7kgfr00qpokgz8w/Feynman_with_units.tar.gz?dl=0",
    "bonus_with_units.tar.gz": "https://www.dropbox.com/s/6sbf36jdllbd3ah/bonus_with_units.tar.gz?dl=0",
    "Feynman_without_units.tar.gz": "https://www.dropbox.com/s/9i05v6yw1kbkup3/Feynman_without_units.tar.gz?dl=0",
    "bonus_without_units.tar.gz": "https://www.dropbox.com/s/o76r2hq8ppp97n8/bonus_without_units.tar.gz?dl=0",
    "FeynmanEquations.csv": "https://space.mit.edu/home/tegmark/aifeynman/FeynmanEquations.csv",
    "FeynmanEquationsDimensionless.csv": "https://space.mit.edu/home/tegmark/aifeynman/FeynmanEquationsDimensionless.csv",
    "BonusEquations.csv": "https://space.mit.edu/home/tegmark/aifeynman/BonusEquations.csv",
    "BonusEquationsDimensionless.csv": "https://space.mit.edu/home/tegmark/aifeynman/BonusEquationsDimensionless.csv",
    "units.csv": "https://space.mit.edu/home/tegmark/aifeynman/units.csv",
}

OVERWRITE = False


class LegacySSLAdapter(HTTPAdapter):
    """Custom adapter to allow connections to servers with weak DH keys."""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


def download_file(url, file_path):
    print(f"Downloading {url} to {file_path}")
    if "dropbox" in url:
        subprocess.run(["wget", "-O", file_path, url], check=True)
    else:
        # Use custom SSL adapter to handle MIT server's weak DH parameters
        session = requests.Session()
        session.mount('https://', LegacySSLAdapter())
        response = session.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)


if __name__ == "__main__":
    data_dir = os.path.join(PROJECT_ROOT, "data", "ai_feyn")
    os.makedirs(data_dir, exist_ok=True)

    # Download files
    print(f"-- Start downloading files to {data_dir}")
    for filename, url in FILE_URLS.items():
        destination_path = os.path.join(data_dir, filename)
        # if file already exists, skip
        if os.path.exists(destination_path) and not OVERWRITE:
            print(f"-- Skipping {filename} (already exists)")
            continue
        download_file(url, destination_path)

    # Untar tar.gz files
    print(f"-- Start untarring files in {data_dir}")
    for filename in FILE_URLS.keys():
        if filename.endswith(".tar.gz"):
            file_path = os.path.join(data_dir, filename)
            dest_path = os.path.dirname(file_path)
            print(f"Untarring {file_path}")
            subprocess.run(["tar", "-xzf", file_path, "-C", dest_path], check=True)

    print(f"-- Done!")
