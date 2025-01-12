import os

def find_file(filename, search_path):
    result = []
    # Walking through directory
    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result

if __name__ == "__main__":
    # Search for dog.png in the project directory
    project_root = "D:/Uni/Đồ án tốt nghiệp/Counting Argorithm"
    file_to_find = "dog.png"
    
    print(f"Searching for {file_to_find}...")
    found_paths = find_file(file_to_find, project_root)
    
    if found_paths:
        print("\nFile found at:")
        for path in found_paths:
            print(path)
    else:
        print(f"\nFile {file_to_find} not found in {project_root}")
