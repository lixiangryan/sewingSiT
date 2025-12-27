import os
import requests
import zipfile
import argparse
from tqdm import tqdm

URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip"
}

def download_file(url, target_dir):
    local_filename = os.path.join(target_dir, url.split('/')[-1])
    if os.path.exists(local_filename):
        print(f"檔案已存在: {local_filename}，跳過下載。")
        return local_filename
        
    print(f"正在下載 {url} 到 {local_filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024 * 1024 # 1MB
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(local_filename, 'wb') as f:
                for data in r.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
    except Exception as e:
        print(f"下載失敗 {url}: {e}")
        if os.path.exists(local_filename):
            os.remove(local_filename)
        return None
        
    return local_filename

def extract_file(zip_path, extract_to):
    if zip_path is None: return
    print(f"正在解壓縮 {zip_path}... (這會調用 NPU/CPU 進行運算)")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("解壓縮完成！")
        # 這裡執行自動刪除
        print(f"正在移除壓縮檔以節省空間: {zip_path}")
        os.remove(zip_path)
    except zipfile.BadZipFile:
        print(f"錯誤: {zip_path} 是損壞的壓縮檔。")

def main(target_dir, split):
    # 路徑優化：將路徑轉換為當前作業系統的絕對路徑
    # 這能有效避免 WSL 裡面 /mnt/f/mnt/f 的邏輯錯誤
    abs_target_dir = os.path.abspath(os.path.expanduser(target_dir))
    
    if not os.path.exists(abs_target_dir):
        os.makedirs(abs_target_dir)

    to_download = []
    if split == 'all':
        to_download = ['val2017', 'annotations', 'train2017']
    elif split == 'test_small':
        to_download = ['val2017', 'annotations']
    elif split == 'train_only':
        to_download = ['train2017']
    else:
        to_download = [split]

    print(f"目標目錄: {abs_target_dir}")
    print(f"下載目標: {to_download}")

    for key in to_download:
        if key not in URLS: continue
        url = URLS[key]
        zip_path = download_file(url, abs_target_dir)
        if zip_path:
            extract_file(zip_path, abs_target_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True, help="資料存放的路徑")
    parser.add_argument("--split", type=str, default="test_small", choices=['all', 'test_small', 'train_only', 'train2017', 'val2017', 'annotations'])
    args = parser.parse_args()
    main(args.target_dir, args.split)