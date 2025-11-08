import os
import subprocess

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

OVERWRITE = True


def download_file(url, file_path):
    command_args = []
    if "dropbox" in url:
        command_args += ["wget", "-O", file_path, url]
    else:
        command_args += ["curl", "--tlsv1.2", "--tls-max", "1.2", "-o", file_path, url]
    print(f"Downloading {url} to {file_path}")
    subprocess.run(command_args, check=True)


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
