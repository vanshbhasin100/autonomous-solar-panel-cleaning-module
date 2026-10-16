import os
import site

patched = False
for site_dir in site.getsitepackages():
    target_file = os.path.join(site_dir, "labelImg", "labelImg.py")
    if os.path.exists(target_file):
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Patch zoom and scroll bar setValue calls that pass floats
        content = content.replace(
            "self.zoom_widget.setValue(value)",
            "self.zoom_widget.setValue(int(value))"
        )
        content = content.replace(
            "bar.setValue(bar.value() + bar.singleStep() * units)",
            "bar.setValue(int(bar.value() + bar.singleStep() * units))"
        )
        
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"Successfully patched {target_file}!")
        patched = True
        break

if not patched:
    print("Could not find labelImg.py to patch.")