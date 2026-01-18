#!/usr/bin/env python3
"""
Inspect and visualize parameters from digit_v4.usd file.

This script displays:
- All prims (bodies/links)
- Mass properties
- Joint properties
- Joint limits
- Drive properties (damping, stiffness)
- Transform information
"""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

from pathlib import Path
from pxr import Usd, UsdPhysics, UsdGeom, Sdf
import omni.usd

# Path
DIGIT_USD_V4 = Path("scripts/tools/digit_assets/digit_v4.usd")


class OutputWriter:
    """Helper class to write to both console and file."""
    def __init__(self, file_path=None):
        self.file = open(file_path, 'w') if file_path else None
        self.file_path = file_path
    
    def write(self, text):
        """Write to both console and file."""
        print(text, end='')
        if self.file:
            self.file.write(text)
    
    def print(self, text=''):
        """Print with newline to both console and file."""
        print(text)
        if self.file:
            self.file.write(text + '\n')
    
    def close(self):
        """Close the file."""
        if self.file:
            self.file.close()
            print(f"\n✓ Output saved to: {self.file_path}")

def print_section(writer, title, width=80):
    """Print a formatted section header."""
    writer.print("\n" + "=" * width)
    writer.print(title.center(width))
    writer.print("=" * width)


def inspect_usd_file(usd_path, output_file=None):
    """Inspect and display all parameters from USD file."""
    writer = OutputWriter(output_file)
    
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        writer.print(f"❌ Failed to open USD file: {usd_path}")
        writer.close()
        return
    
    print_section(writer, f"INSPECTING: {usd_path}")
    
    # Get root prim
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        # Try to find the root
        for prim in stage.Traverse():
            if prim.GetParent().GetPath() == stage.GetPseudoRoot().GetPath():
                root_prim = prim
                break
    
    if root_prim:
        writer.print(f"Root Prim: {root_prim.GetPath()}")
        writer.print(f"Type: {root_prim.GetTypeName()}")
    
    # Collect all prims
    all_prims = list(stage.Traverse())
    writer.print(f"\nTotal Prims: {len(all_prims)}")
    
    # Separate prims by type
    rigid_bodies = []
    joints = []
    meshes = []
    xforms = []
    
    for prim in all_prims:
        if UsdPhysics.RigidBodyAPI(prim):
            rigid_bodies.append(prim)
        
        # Check for joints by looking for joint-related attributes
        has_joint = False
        attrs = prim.GetAttributes()
        for attr in attrs:
            attr_name = attr.GetName()
            # Check for common joint attribute patterns
            if any(keyword in attr_name.lower() for keyword in ['limit', 'drive', 'damping', 'stiffness', 'maxforce', 'joint']):
                has_joint = True
                break
        
        if has_joint:
            joints.append(prim)
        
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
        if prim.IsA(UsdGeom.Xform):
            xforms.append(prim)
    
    writer.print(f"Rigid Bodies: {len(rigid_bodies)}")
    writer.print(f"Joints: {len(joints)}")
    writer.print(f"Meshes: {len(meshes)}")
    writer.print(f"Xforms: {len(xforms)}")
    
    # Display Rigid Bodies
    print_section(writer, "RIGID BODIES (Links)")
    for i, prim in enumerate(rigid_bodies, 1):
        writer.print(f"\n[{i}] {prim.GetPath()}")
        writer.print(f"    Name: {prim.GetName()}")
        writer.print(f"    Type: {prim.GetTypeName()}")
        
        # Get mass properties
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        mass_api = UsdPhysics.MassAPI(rigid_body_api)
        
        if mass_api:
            mass_attr = mass_api.GetMassAttr()
            if mass_attr and mass_attr.HasAuthoredValue():
                mass = mass_attr.Get()
                writer.print(f"    Mass: {mass:.6f} kg")
            
            com_attr = mass_api.GetCenterOfMassAttr()
            if com_attr and com_attr.HasAuthoredValue():
                com = com_attr.Get()
                writer.print(f"    Center of Mass: ({com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f})")
            
            diaginertia_attr = mass_api.GetDiagonalInertiaAttr()
            if diaginertia_attr and diaginertia_attr.HasAuthoredValue():
                inertia = diaginertia_attr.Get()
                writer.print(f"    Diagonal Inertia: [{inertia[0]:.6f}, {inertia[1]:.6f}, {inertia[2]:.6f}]")
        
        # Get transform
        xform = UsdGeom.Xformable(prim)
        if xform:
            ops = xform.GetOrderedXformOps()
            if ops:
                writer.print(f"    Transform Ops: {len(ops)}")
                for op in ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        val = op.Get()
                        writer.print(f"      Translate: ({val[0]:.4f}, {val[1]:.4f}, {val[2]:.4f})")
                    elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                        val = op.Get()
                        writer.print(f"      RotateXYZ: ({val[0]:.4f}, {val[1]:.4f}, {val[2]:.4f})")
                    elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        val = op.Get()
                        writer.print(f"      Scale: ({val[0]:.4f}, {val[1]:.4f}, {val[2]:.4f})")
    
    # Display Joints
    print_section(writer, "JOINTS")
    for i, prim in enumerate(joints, 1):
        writer.print(f"\n[{i}] {prim.GetPath()}")
        writer.print(f"    Name: {prim.GetName()}")
        writer.print(f"    Type: {prim.GetTypeName()}")
        
        # Get joint limits by checking attributes directly
        low = None
        high = None
        for attr in prim.GetAttributes():
            attr_name = attr.GetName()
            if 'low' in attr_name.lower() and attr.HasAuthoredValue():
                try:
                    low = attr.Get()
                    writer.print(f"    Lower Limit: {low:.6f} rad")
                except:
                    pass
            if 'high' in attr_name.lower() and attr.HasAuthoredValue():
                try:
                    high = attr.Get()
                    writer.print(f"    Upper Limit: {high:.6f} rad")
                except:
                    pass
        
        if low is not None and high is not None:
            writer.print(f"    Range: [{low:.6f}, {high:.6f}] rad")
        
        # Get drive properties by checking attributes directly
        for attr in prim.GetAttributes():
            attr_name = attr.GetName().lower()
            if attr.HasAuthoredValue():
                try:
                    value = attr.Get()
                    if 'damping' in attr_name:
                        writer.print(f"    Damping ({attr.GetName()}): {value:.6f}")
                    elif 'stiffness' in attr_name:
                        writer.print(f"    Stiffness ({attr.GetName()}): {value:.6f}")
                    elif 'maxforce' in attr_name or 'max_force' in attr_name:
                        writer.print(f"    Max Force ({attr.GetName()}): {value:.6f}")
                except:
                    pass
        
        # Get joint DOF
        try:
            joint_dof = UsdPhysics.JointDOF(prim)
            if joint_dof:
                writer.print(f"    Joint DOF: Present")
        except:
            pass
    
    # Display hierarchy
    print_section(writer, "PRIM HIERARCHY")
    def print_hierarchy(prim, indent=0):
        prefix = "  " * indent + "├─ " if indent > 0 else ""
        writer.print(f"{prefix}{prim.GetName()} ({prim.GetTypeName()})")
        
        for child in prim.GetChildren():
            print_hierarchy(child, indent + 1)
    
    if root_prim:
        print_hierarchy(root_prim)
    
    # Summary
    print_section(writer, "SUMMARY")
    writer.print(f"Total Prims: {len(all_prims)}")
    writer.print(f"Rigid Bodies: {len(rigid_bodies)}")
    writer.print(f"Joints: {len(joints)}")
    writer.print(f"Meshes: {len(meshes)}")
    
    # List all joint names
    if joints:
        writer.print(f"\nJoint Names ({len(joints)}):")
        for prim in joints:
            writer.print(f"  - {prim.GetName()}")
    
    # List all body names
    if rigid_bodies:
        writer.print(f"\nBody Names ({len(rigid_bodies)}):")
        for prim in rigid_bodies:
            writer.print(f"  - {prim.GetName()}")
    
    writer.close()


def main():
    if not DIGIT_USD_V4.exists():
        print(f"❌ USD file not found: {DIGIT_USD_V4}")
        print(f"   Please ensure the file exists at: {DIGIT_USD_V4.absolute()}")
        return
    
    # Output file path
    output_file = "scripts/tools/digit_v4_parameters.txt"
    inspect_usd_file(DIGIT_USD_V4, output_file)


if __name__ == "__main__":
    main()
    simulation_app.close()

