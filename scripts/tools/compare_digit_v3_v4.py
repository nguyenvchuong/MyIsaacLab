#!/usr/bin/env python3
"""
Compare parameters between digit_v3 (MJCF) and digit_v4 (USD).

This script extracts:
- Mass values
- Inertia tensors
- Joint limits (ranges)
- Joint damping
- Joint armature
- Initial positions
"""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import xml.etree.ElementTree as ET
from pathlib import Path
from pxr import Usd, UsdPhysics, UsdGeom, Gf
import re

# Paths
DIGIT_XML_V3 = Path("../mujoco_menagerie/Digit_v3/digit.xml")
DIGIT_USD_V4 = Path("scripts/tools/digit_assets/digit_v4.usd")


def extract_mjcf_parameters(xml_path):
    """Extract parameters from MJCF XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    params = {
        'bodies': {},
        'joints': {},
        'actuators': {}
    }
    
    # Extract body masses and inertias
    for body in root.findall('.//body'):
        body_name = body.get('name')
        if not body_name:
            continue
            
        inertial = body.find('inertial')
        if inertial is not None:
            mass = inertial.get('mass')
            diaginertia = inertial.get('diaginertia')
            pos = inertial.get('pos', '0 0 0')
            quat = inertial.get('quat', '1 0 0 0')
            
            params['bodies'][body_name] = {
                'mass': float(mass) if mass else None,
                'diaginertia': [float(x) for x in diaginertia.split()] if diaginertia else None,
                'pos': [float(x) for x in pos.split()],
                'quat': [float(x) for x in quat.split()]
            }
    
    # Extract joint parameters
    for body in root.findall('.//body'):
        for joint in body.findall('joint'):
            joint_name = joint.get('name')
            if not joint_name:
                continue
                
            range_attr = joint.get('range')
            damping = joint.get('damping', '0')
            frictionloss = joint.get('frictionloss', '0')
            armature = joint.get('armature', '0')
            axis = joint.get('axis', '0 0 1')
            joint_type = joint.get('type', 'hinge')
            
            params['joints'][joint_name] = {
                'range': [float(x) for x in range_attr.split()] if range_attr else None,
                'damping': float(damping),
                'frictionloss': float(frictionloss),
                'armature': float(armature),
                'axis': [float(x) for x in axis.split()],
                'type': joint_type
            }
    
    # Extract actuator parameters
    for motor in root.findall('.//motor'):
        motor_name = motor.get('name')
        joint = motor.get('joint')
        ctrlrange = motor.get('ctrlrange')
        
        if motor_name and joint:
            params['actuators'][motor_name] = {
                'joint': joint,
                'ctrlrange': [float(x) for x in ctrlrange.split()] if ctrlrange else None
            }
    
    return params


def extract_usd_parameters(usd_path):
    """Extract parameters from USD file."""
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open USD file: {usd_path}")
        return None
    
    params = {
        'bodies': {},
        'joints': {},
        'root_pos': None
    }
    
    # Find the root prim (usually the robot name)
    root_prim = None
    for prim in stage.Traverse():
        # Check if this prim has RigidBodyAPI
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        if rigid_body_api:
            # Check if this is the root
            if prim.GetParent().GetPath() == prim.GetStage().GetPseudoRoot().GetPath():
                root_prim = prim
                break
    
    # If no root found, try common names
    if not root_prim:
        for path in ['/digit', '/Digit', '/World/Digit', '/World/digit']:
            prim = stage.GetPrimAtPath(path)
            if prim:
                rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
                if rigid_body_api:
                    root_prim = prim
                    break
    
    if root_prim:
        # Get root position
        xform = UsdGeom.Xformable(root_prim)
        if xform:
            translate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op:
                params['root_pos'] = list(translate_op.Get())
    
    # Extract body masses and properties
    for prim in stage.Traverse():
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        if rigid_body_api:
            body_name = prim.GetName()
            
            # Get mass (MassAPI is accessed through RigidBodyAPI)
            mass_api = UsdPhysics.MassAPI(rigid_body_api)
            mass = None
            if mass_api:
                mass_attr = mass_api.GetMassAttr()
                if mass_attr and mass_attr.HasAuthoredValue():
                    mass = mass_attr.Get()
            
            # Get center of mass
            com = None
            if mass_api:
                com_attr = mass_api.GetCenterOfMassAttr()
                if com_attr and com_attr.HasAuthoredValue():
                    com = list(com_attr.Get())
            
            # Get inertia (diagonal)
            diaginertia = None
            if mass_api:
                diaginertia_attr = mass_api.GetDiagonalInertiaAttr()
                if diaginertia_attr and diaginertia_attr.HasAuthoredValue():
                    diaginertia = list(diaginertia_attr.Get())
            
            if mass or diaginertia:
                params['bodies'][body_name] = {
                    'mass': mass,
                    'diaginertia': diaginertia,
                    'center_of_mass': com
                }
    
    # Extract joint parameters
    # Look for prims with joint-related schemas
    for prim in stage.Traverse():
        # Check if this prim has joint-related attributes
        # Joints in USD typically have DriveAPI or LimitAPI
        has_joint_data = False
        joint_name = prim.GetName()
        
        # Try to get joint limits using LimitAPI
        low = None
        high = None
        try:
            limit_api = UsdPhysics.LimitAPI(prim)
            if limit_api:
                low_attr = limit_api.GetLowAttr()
                high_attr = limit_api.GetHighAttr()
                if low_attr and low_attr.HasAuthoredValue():
                    low = low_attr.Get()
                    has_joint_data = True
                if high_attr and high_attr.HasAuthoredValue():
                    high = high_attr.Get()
                    has_joint_data = True
        except:
            pass
        
        # Try to get damping using DriveAPI
        damping = None
        try:
            # Try angular drive first
            drive_api = UsdPhysics.DriveAPI(prim, "angular")
            if drive_api:
                damping_attr = drive_api.GetDampingAttr()
                if damping_attr and damping_attr.HasAuthoredValue():
                    damping = damping_attr.Get()
                    has_joint_data = True
            else:
                # Try linear drive
                drive_api = UsdPhysics.DriveAPI(prim, "linear")
                if drive_api:
                    damping_attr = drive_api.GetDampingAttr()
                    if damping_attr and damping_attr.HasAuthoredValue():
                        damping = damping_attr.Get()
                        has_joint_data = True
        except:
            pass
        
        # Also check for joint DOF (Degrees of Freedom) which indicates a joint
        try:
            joint_dof = UsdPhysics.JointDOF(prim)
            if joint_dof:
                has_joint_data = True
        except:
            pass
        
        if has_joint_data:
            params['joints'][joint_name] = {
                'range': [low, high] if low is not None or high is not None else None,
                'damping': damping
            }
    
    return params


def compare_parameters(mjcf_params, usd_params):
    """Compare parameters between MJCF and USD."""
    print("=" * 80)
    print("COMPARISON: Digit V3 (MJCF) vs Digit V4 (USD)")
    print("=" * 80)
    
    # Compare bodies (masses)
    print("\n" + "=" * 80)
    print("BODY MASSES")
    print("=" * 80)
    print(f"{'Body Name':<30} {'V3 Mass (MJCF)':<20} {'V4 Mass (USD)':<20} {'Difference':<15}")
    print("-" * 80)
    
    all_bodies = set(mjcf_params['bodies'].keys()) | set(usd_params['bodies'].keys())
    for body_name in sorted(all_bodies):
        v3_mass = mjcf_params['bodies'].get(body_name, {}).get('mass')
        v4_mass = usd_params['bodies'].get(body_name, {}).get('mass')
        
        v3_str = f"{v3_mass:.4f}" if v3_mass else "N/A"
        v4_str = f"{v4_mass:.4f}" if v4_mass else "N/A"
        
        if v3_mass and v4_mass:
            diff = abs(v3_mass - v4_mass)
            diff_str = f"{diff:.4f}"
        else:
            diff_str = "N/A"
        
        print(f"{body_name:<30} {v3_str:<20} {v4_str:<20} {diff_str:<15}")
    
    # Compare joints
    print("\n" + "=" * 80)
    print("JOINT PARAMETERS")
    print("=" * 80)
    print(f"{'Joint Name':<35} {'V3 Range':<25} {'V4 Range':<25} {'V3 Damping':<15} {'V4 Damping':<15}")
    print("-" * 80)
    
    all_joints = set(mjcf_params['joints'].keys()) | set(usd_params['joints'].keys())
    for joint_name in sorted(all_joints):
        v3_joint = mjcf_params['joints'].get(joint_name, {})
        v4_joint = usd_params['joints'].get(joint_name, {})
        
        v3_range = v3_joint.get('range')
        v4_range = v4_joint.get('range')
        
        v3_range_str = f"[{v3_range[0]:.3f}, {v3_range[1]:.3f}]" if v3_range else "N/A"
        v4_range_str = f"[{v4_range[0]:.3f}, {v4_range[1]:.3f}]" if v4_range else "N/A"
        
        v3_damping = v3_joint.get('damping', 0)
        v4_damping = v4_joint.get('damping')
        
        v3_damp_str = f"{v3_damping:.3f}" if v3_damping else "N/A"
        v4_damp_str = f"{v4_damping:.3f}" if v4_damping else "N/A"
        
        print(f"{joint_name:<35} {v3_range_str:<25} {v4_range_str:<25} {v3_damp_str:<15} {v4_damp_str:<15}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"V3 Bodies: {len(mjcf_params['bodies'])}")
    print(f"V4 Bodies: {len(usd_params['bodies'])}")
    print(f"V3 Joints: {len(mjcf_params['joints'])}")
    print(f"V4 Joints: {len(usd_params['joints'])}")
    print(f"V3 Actuators: {len(mjcf_params['actuators'])}")


def main():
    print("Extracting parameters from digit.xml (V3)...")
    mjcf_params = extract_mjcf_parameters(DIGIT_XML_V3)
    
    print("Extracting parameters from digit_v4.usd (V4)...")
    usd_params = extract_usd_parameters(DIGIT_USD_V4)
    
    if usd_params:
        compare_parameters(mjcf_params, usd_params)
    else:
        print("Failed to extract USD parameters")


if __name__ == "__main__":
    main()
    simulation_app.close()

