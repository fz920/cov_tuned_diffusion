#!/usr/bin/env python3
"""
Script to automatically replace hardcoded paths with references to the path configuration system.
This script searches for hardcoded paths containing the username and replaces them with
appropriate calls to the path configuration system.
"""

import os
import re
import glob
import argparse
from pathlib import Path

def update_imports(content):
    """Add necessary imports for path_config if they don't exist."""
    if "from utils.path_config import" not in content:
        # Check if there are any imports at all
        if "import " in content:
            # Find the last import statement
            import_lines = re.findall(r'^.*import.*$', content, re.MULTILINE)
            if import_lines:
                last_import = import_lines[-1]
                # Add our import after the last import
                return content.replace(
                    last_import,
                    last_import + "\n\nfrom utils.path_config import (\n"
                    "    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,\n"
                    "    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR\n"
                    ")"
                )
        # If no imports found, add at the beginning
        return "from utils.path_config import (\n" \
               "    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,\n" \
               "    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR\n" \
               ")\n\n" + content
    return content

def replace_hardcoded_paths(content):
    """Replace hardcoded paths with path configuration calls."""
    # Replace model config paths
    content = re.sub(
        r"f'[^']*?/rds/user/fz287/hpc-work/dissertation/[^']*?/model/configs/([^']*?)_([^']*?)_([^']*?)\.yaml'",
        r"get_config_path('\1', '\2', '\3')",
        content
    )
    content = re.sub(
        r"f'[^']*?/rds/user/fz287/hpc-work/dissertation/[^']*?/model/configs/([^']*?)_([^']*?)\.yaml'",
        r"get_config_path('\1', '\2')",
        content
    )
    
    # Replace model checkpoint paths
    content = re.sub(
        r"f'[^']*?/rds/user/fz287/hpc-work/dissertation/checkpoints/model_checkpoints/([^']*?)/([^']*?)/([^']*?)_([^']*?)_([^']*?)\.pth'",
        r"get_model_checkpoint_path('\1', '\3', '\4', \5)",
        content
    )
    
    # Replace figure paths
    content = re.sub(
        r"f'[^']*?/rds/user/fz287/hpc-work/dissertation/[^']*?/figures/([^']*?)'",
        r"FIGURES_DIR / '\1'",
        content
    )
    
    # Replace sample paths
    content = re.sub(
        r"f'[^']*?/rds/user/fz287/hpc-work/dissertation/checkpoints/samples/([^']*?)'",
        r"CHECKPOINTS_DIR / 'samples' / '\1'",
        content
    )
    
    # Replace parameter paths
    content = re.sub(
        r"f'[^']*?/rds/user/fz287/hpc-work/dissertation/checkpoints/params_checkpoints/([^']*?)'",
        r"CHECKPOINTS_DIR / 'params_checkpoints' / '\1'",
        content
    )
    
    # Replace other checkpoints paths
    content = re.sub(
        r"f'[^']*?/rds/user/fz287/hpc-work/dissertation/checkpoints/([^']*?)'",
        r"CHECKPOINTS_DIR / '\1'",
        content
    )
    
    # General fallback for other paths
    content = re.sub(
        r"str(Path(__file__).parent.parent / '([^')]*?)'",
        r"str(Path(__file__).parent.parent / '\1')",
        content
    )
    
    return content

def process_file(file_path, dry_run=False):
    """Process a single file to replace hardcoded paths."""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Skip if no hardcoded paths
    if "fz287" not in content:
        print(f"  No hardcoded paths found in {file_path}")
        return False
    
    # Update the content
    updated_content = replace_hardcoded_paths(content)
    updated_content = update_imports(updated_content)
    
    # Check if anything changed
    if content == updated_content:
        print(f"  No changes made to {file_path}")
        return False
    
    if dry_run:
        print(f"  Would update {file_path} (dry run)")
        return True
    
    # Write back the updated content
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"  Updated {file_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fix hardcoded paths in the codebase")
    parser.add_argument('--dry-run', action='store_true', help="Don't actually modify files")
    args = parser.parse_args()
    
    # Get the base directory (where this script is)
    base_dir = Path(__file__).parent
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Process each file
    updated_count = 0
    for file_path in python_files:
        if process_file(file_path, args.dry_run):
            updated_count += 1
    
    print(f"\nSummary: {updated_count} files would be updated" if args.dry_run else f"\nSummary: {updated_count} files updated")

if __name__ == "__main__":
    main() 