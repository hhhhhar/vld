import argparse
import re
import os
from pathlib import Path
from xml.etree import ElementTree as ET

VALID_RE = re.compile(r'[^A-Za-z0-9_.]')  # characters to replace with underscore

def sanitize_name(name: str) -> str:
    if name is None:
        return name
    new = VALID_RE.sub('_', name)
    if new and new[0].isdigit():
        new = 'p_' + new
    if not new:
        new = 'prim_unnamed'
    return new

def unique_name(path: Path) -> Path:
    """If path exists, append _1, _2 ... before suffix until unique."""
    if not path.exists():
        return path
    base = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{base}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def build_file_map_from_assets_dir(assets_dir: Path):
    # build absolute path map for files that remain or were renamed in the directory
    file_map = {}
    for p in assets_dir.rglob('*'):
        if p.is_file():
            file_map[str(p.resolve())] = str(p.resolve())
    return file_map

def ensure_inertial(link: ET.Element):
    inertial = link.find('inertial')
    added = False
    if inertial is None:
        inertial = ET.Element('inertial')
        mass = ET.SubElement(inertial, 'mass')
        mass.set('value', '0.001')
        inertia = ET.SubElement(inertial, 'inertia')
        inertia.set('ixx', '1e-6')
        inertia.set('iyy', '1e-6')
        inertia.set('izz', '1e-6')
        link.insert(0, inertial)
        added = True
    else:
        mass = inertial.find('mass')
        if mass is None or not mass.get('value'):
            if mass is None:
                mass = ET.SubElement(inertial, 'mass')
            mass.set('value', '0.001')
            added = True
    return added

def sanitize_urdf_and_update_meshes(in_urdf: Path, out_urdf: Path, assets_dir: Path, file_map: dict):
    tree = ET.parse(in_urdf)
    root = tree.getroot()
    name_map = {}

    # sanitize link names and ensure inertial mass
    for link in root.findall('link'):
        old = link.get('name')
        new = sanitize_name(old)
        if new != old:
            link.set('name', new)
            name_map[old] = new
        if ensure_inertial(link):
            print(f"Added default inertial to link: {link.get('name')}")

    # sanitize joints and update parent/child references
    for joint in root.findall('joint'):
        old = joint.get('name')
        if old:
            new = sanitize_name(old)
            if new != old:
                joint.set('name', new)
                name_map[old] = new
        parent = joint.find('parent')
        child = joint.find('child')
        for elem in (parent, child):
            if elem is not None:
                pname = elem.get('link')
                if pname in name_map:
                    elem.set('link', name_map[pname])
                else:
                    elem.set('link', sanitize_name(pname))

    # update mesh filenames
    missing_files = []
    updated = 0
    for mesh in root.findall('.//mesh'):
        fn = mesh.get('filename')
        if not fn or fn.strip() == '':
            print("Warning: mesh with empty filename found in URDF; consider fixing this mesh entry.")
            continue
        # handle package:// or absolute or relative paths - try to map by basename first
        basename = os.path.basename(fn)
        replaced = False
        for orig_abs, new_abs in file_map.items():
            if os.path.basename(orig_abs) == basename:
                # compute a relative path from assets_dir to the new_abs if original fn was relative inside assets_dir
                try:
                    new_rel = os.path.relpath(new_abs, start=str(assets_dir))
                except Exception:
                    new_rel = os.path.basename(new_abs)
                dirpart = os.path.dirname(fn)
                if dirpart and dirpart not in ('.', ''):
                    candidate = os.path.join(dirpart, os.path.basename(new_rel)).replace('\\', '/')
                else:
                    candidate = new_rel.replace('\\', '/')
                mesh.set('filename', candidate)
                updated += 1
                replaced = True
                break
        if not replaced:
            # as a last resort, try to find by basename in assets_dir
            cand = None
            for p in Path(assets_dir).rglob(basename):
                cand = p
                break
            if cand is not None:
                try:
                    new_rel = os.path.relpath(str(cand.resolve()), start=str(assets_dir))
                except Exception:
                    new_rel = cand.name
                mesh.set('filename', new_rel.replace('\\', '/'))
                updated += 1
            else:
                missing_files.append(fn)

    if updated:
        print(f"Updated {updated} mesh filename(s) in URDF to point to renamed assets.")

    if missing_files:
        print("Warning: the following mesh filenames could not be located/updated (may be external or missing):")
        for m in missing_files:
            print("  ", m)

    # sanitize collision/visual <name> children (if present)
    for tag in root.findall('.//collision') + root.findall('.//visual'):
        name_child = tag.find('name')
        if name_child is not None and name_child.text:
            name_child.text = sanitize_name(name_child.text)

    # final pass: apply name_map to attributes that reference link names
    for old, new in name_map.items():
        for elem in root.findall(".//*[@link='%s']" % old):
            elem.set('link', new)

    tree.write(out_urdf, encoding='utf-8', xml_declaration=True)
    print("Wrote sanitized URDF to:", out_urdf)

def main():
    parser = argparse.ArgumentParser(description='Sanitize assets and URDF for Isaac Sim / USD import')
    parser.add_argument('--assets-dir', default="/home/hhhar/liuliu/dige/47648/textured_objs", help='Path to directory containing asset files (meshes/textures)')
    parser.add_argument('--input-urdf', default="/home/hhhar/liuliu/dige/47648/mobility.urdf", help='Input URDF file path')
    parser.add_argument('--output-urdf', default="/home/hhhar/liuliu/dige/47648/renamed.urdf", help='Output sanitized URDF file path')
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    in_urdf = Path(args.input_urdf)
    out_urdf = Path(args.output_urdf)

    if not assets_dir.exists() or not assets_dir.is_dir():
        print("Error: assets-dir does not exist or is not a directory:", assets_dir)
        return
    if not in_urdf.exists():
        print("Error: input URDF does not exist:", in_urdf)
        return

    print("Renaming assets in:", assets_dir)
    # rename assets and get mapping of original->new (absolute paths)
    # Note: we operate in-place; if you need originals, copy first!
    original_files = {str(p.resolve()): p for p in assets_dir.rglob('*') if p.is_file()}
    mapping = {}
    for orig_abs, p in sorted(original_files.items()):
        sanitized = sanitize_name(p.name)
        if sanitized == p.name:
            mapping[orig_abs] = orig_abs  # unchanged
            continue
        new_path = p.with_name(sanitized)
        if new_path.exists():
            new_path = unique_name(new_path)
        p.rename(new_path)
        mapping[orig_abs] = str(new_path.resolve())
        print(f"Renamed: {p.name} -> {new_path.name}")

    # After renames, build file_map of actual assets (absolute -> absolute)
    file_map = build_file_map_from_assets_dir(assets_dir)
    # incorporate mapping where keys were original abs paths
    for k,v in mapping.items():
        # ensure mapping present (orig->new) for lookup by basename
        file_map[k] = v

    sanitize_urdf_and_update_meshes(in_urdf, out_urdf, assets_dir, file_map)
    print("Done. Please inspect the output URDF and run Isaac Sim importer on the sanitized files.")

if __name__ == '__main__':
    main()