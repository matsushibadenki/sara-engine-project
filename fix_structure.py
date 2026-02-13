import os
import shutil

def fix_project_structure():
    print("Fixing project structure...")
    
    # 1. lib.rs の場所を探して src/lib.rs に移動
    target_path = os.path.join("src", "lib.rs")
    possible_paths = [
        os.path.join("sara_rust_core", "src", "lib.rs"), # 元の場所
        os.path.join("src", "sara_rust_core", "src", "lib.rs"),
        os.path.join("src", "sara_engine", "lib.rs"),
        "lib.rs"
    ]
    
    found = False
    if os.path.exists(target_path):
        print(f"✅ {target_path} already exists.")
        found = True
    else:
        for p in possible_paths:
            if os.path.exists(p):
                print(f"Found lib.rs at {p}. Moving to {target_path}...")
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.move(p, target_path)
                found = True
                
                # 元のディレクトリが空なら削除
                old_dir = os.path.dirname(os.path.dirname(p))
                if "sara_rust_core" in old_dir and os.path.exists(old_dir):
                    try:
                        shutil.rmtree(old_dir)
                        print(f"Removed old directory: {old_dir}")
                    except:
                        pass
                break
    
    if not found:
        print("❌ Error: lib.rs not found anywhere. Please check your files.")
        return

    # 2. Cargo.toml の path を明示的に修正
    cargo_path = "Cargo.toml"
    if os.path.exists(cargo_path):
        with open(cargo_path, "r") as f:
            content = f.read()
        
        # path = "src/lib.rs" がなければ追加
        if 'path = "src/lib.rs"' not in content:
            print("Updating Cargo.toml with explicit path...")
            new_content = content.replace(
                'crate-type = ["cdylib"]', 
                'crate-type = ["cdylib"]\npath = "src/lib.rs"'
            )
            with open(cargo_path, "w") as f:
                f.write(new_content)
        else:
            print("✅ Cargo.toml is already correct.")
    else:
        print("❌ Error: Cargo.toml not found in root.")

if __name__ == "__main__":
    fix_project_structure()
