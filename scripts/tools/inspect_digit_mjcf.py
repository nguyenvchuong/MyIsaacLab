#!/usr/bin/env python3
"""
Inspect and visualize parameters from digit.xml (MJCF) file.

This script displays:
- All bodies (links)
- Mass properties
- Inertia tensors
- Joint parameters
- Joint limits (ranges)
- Joint damping, friction, armature
- Actuator parameters
- Initial positions
"""

import xml.etree.ElementTree as ET
from pathlib import Path

# Path
DIGIT_XML_V3 = Path("../mujoco_menagerie/Digit_v3/digit.xml")


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


def inspect_mjcf_file(xml_path, output_file=None):
    """Inspect and display all parameters from MJCF XML file."""
    writer = OutputWriter(output_file)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    print_section(writer, f"INSPECTING: {xml_path}")
    
    model_name = root.get('model', 'unknown')
    writer.print(f"Model Name: {model_name}")
    
    # Get compiler settings
    compiler = root.find('compiler')
    if compiler is not None:
        writer.print(f"\nCompiler Settings:")
        for key, value in compiler.attrib.items():
            writer.print(f"  {key}: {value}")
    
    # Collect all bodies
    all_bodies = root.findall('.//body')
    writer.print(f"\nTotal Bodies: {len(all_bodies)}")
    
    # Extract body information
    bodies_info = []
    for body in all_bodies:
        body_name = body.get('name')
        if not body_name:
            continue
        
        body_info = {
            'name': body_name,
            'pos': body.get('pos', '0 0 0'),
            'quat': body.get('quat', '1 0 0 0'),
            'mass': None,
            'inertia': None,
            'com_pos': None,
            'com_quat': None
        }
        
        # Get inertial properties
        inertial = body.find('inertial')
        if inertial is not None:
            mass = inertial.get('mass')
            diaginertia = inertial.get('diaginertia')
            pos = inertial.get('pos', '0 0 0')
            quat = inertial.get('quat', '1 0 0 0')
            
            body_info['mass'] = float(mass) if mass else None
            body_info['inertia'] = [float(x) for x in diaginertia.split()] if diaginertia else None
            body_info['com_pos'] = [float(x) for x in pos.split()]
            body_info['com_quat'] = [float(x) for x in quat.split()]
        
        bodies_info.append(body_info)
    
    # Extract joint information
    joints_info = []
    for body in all_bodies:
        for joint in body.findall('joint'):
            joint_name = joint.get('name')
            if not joint_name:
                continue
            
            joint_info = {
                'name': joint_name,
                'type': joint.get('type', 'hinge'),
                'pos': joint.get('pos', '0 0 0'),
                'axis': joint.get('axis', '0 0 1'),
                'range': None,
                'damping': None,
                'frictionloss': None,
                'armature': None,
                'stiffness': None,
                'limited': joint.get('limited', 'false')
            }
            
            range_attr = joint.get('range')
            if range_attr:
                joint_info['range'] = [float(x) for x in range_attr.split()]
            
            damping = joint.get('damping')
            if damping:
                joint_info['damping'] = float(damping)
            
            frictionloss = joint.get('frictionloss')
            if frictionloss:
                joint_info['frictionloss'] = float(frictionloss)
            
            armature = joint.get('armature')
            if armature:
                joint_info['armature'] = float(armature)
            
            stiffness = joint.get('stiffness')
            if stiffness:
                joint_info['stiffness'] = float(stiffness)
            
            joints_info.append(joint_info)
    
    # Extract actuator information
    actuators_info = []
    for motor in root.findall('.//motor'):
        motor_name = motor.get('name')
        joint = motor.get('joint')
        ctrlrange = motor.get('ctrlrange')
        gear = motor.get('gear')
        ctrllimited = motor.get('ctrllimited', 'false')
        
        if motor_name and joint:
            actuator_info = {
                'name': motor_name,
                'joint': joint,
                'ctrlrange': [float(x) for x in ctrlrange.split()] if ctrlrange else None,
                'gear': float(gear) if gear else None,
                'ctrllimited': ctrllimited
            }
            actuators_info.append(actuator_info)
    
    # Extract mesh information
    meshes_info = []
    asset = root.find('asset')
    if asset is not None:
        for mesh in asset.findall('mesh'):
            mesh_name = mesh.get('name')
            mesh_file = mesh.get('file')
            scale = mesh.get('scale', '1 1 1')
            
            if mesh_name:
                meshes_info.append({
                    'name': mesh_name,
                    'file': mesh_file,
                    'scale': [float(x) for x in scale.split()]
                })
    
    # Display Bodies
    print_section(writer, "BODIES (Links)")
    for i, body_info in enumerate(bodies_info, 1):
        writer.print(f"\n[{i}] {body_info['name']}")
        writer.print(f"    Position: ({body_info['pos']})")
        writer.print(f"    Quaternion: ({body_info['quat']})")
        
        if body_info['mass'] is not None:
            writer.print(f"    Mass: {body_info['mass']:.6f} kg")
        
        if body_info['inertia'] is not None:
            inertia = body_info['inertia']
            writer.print(f"    Diagonal Inertia: [{inertia[0]:.6f}, {inertia[1]:.6f}, {inertia[2]:.6f}]")
        
        if body_info['com_pos']:
            com = body_info['com_pos']
            writer.print(f"    Center of Mass: ({com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f})")
        
        if body_info['com_quat']:
            com_quat = body_info['com_quat']
            writer.print(f"    COM Quaternion: ({com_quat[0]:.6f}, {com_quat[1]:.6f}, {com_quat[2]:.6f}, {com_quat[3]:.6f})")
    
    # Display Joints
    print_section(writer, "JOINTS")
    for i, joint_info in enumerate(joints_info, 1):
        writer.print(f"\n[{i}] {joint_info['name']}")
        writer.print(f"    Type: {joint_info['type']}")
        writer.print(f"    Position: ({joint_info['pos']})")
        writer.print(f"    Axis: ({joint_info['axis']})")
        writer.print(f"    Limited: {joint_info['limited']}")
        
        if joint_info['range']:
            r = joint_info['range']
            writer.print(f"    Range: [{r[0]:.6f}, {r[1]:.6f}] rad")
        
        if joint_info['damping'] is not None:
            writer.print(f"    Damping: {joint_info['damping']:.6f}")
        
        if joint_info['frictionloss'] is not None:
            writer.print(f"    Friction Loss: {joint_info['frictionloss']:.6f}")
        
        if joint_info['armature'] is not None:
            writer.print(f"    Armature: {joint_info['armature']:.6f}")
        
        if joint_info['stiffness'] is not None:
            writer.print(f"    Stiffness: {joint_info['stiffness']:.6f}")
    
    # Display Actuators
    print_section(writer, "ACTUATORS (Motors)")
    for i, actuator_info in enumerate(actuators_info, 1):
        writer.print(f"\n[{i}] {actuator_info['name']}")
        writer.print(f"    Joint: {actuator_info['joint']}")
        
        if actuator_info['ctrlrange']:
            r = actuator_info['ctrlrange']
            writer.print(f"    Control Range: [{r[0]:.6f}, {r[1]:.6f}]")
        
        if actuator_info['gear'] is not None:
            writer.print(f"    Gear: {actuator_info['gear']:.6f}")
        
        writer.print(f"    Control Limited: {actuator_info['ctrllimited']}")
    
    # Display Meshes
    if meshes_info:
        print_section(writer, "MESHES")
        writer.print(f"Total Meshes: {len(meshes_info)}")
        for i, mesh_info in enumerate(meshes_info, 1):
            writer.print(f"\n[{i}] {mesh_info['name']}")
            writer.print(f"    File: {mesh_info['file']}")
            if mesh_info['scale'] != [1.0, 1.0, 1.0]:
                s = mesh_info['scale']
                writer.print(f"    Scale: ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f})")
    
    # Display Hierarchy
    print_section(writer, "BODY HIERARCHY")
    def print_hierarchy(body, indent=0, writer=writer):
        body_name = body.get('name', 'unnamed')
        body_type = body.get('type', 'body')
        prefix = "  " * indent + "├─ " if indent > 0 else ""
        writer.print(f"{prefix}{body_name} ({body_type})")
        
        for child in body.findall('body'):
            print_hierarchy(child, indent + 1, writer)
    
    worldbody = root.find('worldbody')
    if worldbody is not None:
        for body in worldbody.findall('body'):
            print_hierarchy(body, 0, writer)
    
    # Summary
    print_section(writer, "SUMMARY")
    writer.print(f"Total Bodies: {len(bodies_info)}")
    writer.print(f"Total Joints: {len(joints_info)}")
    writer.print(f"Total Actuators: {len(actuators_info)}")
    writer.print(f"Total Meshes: {len(meshes_info)}")
    
    # List all joint names
    if joints_info:
        writer.print(f"\nJoint Names ({len(joints_info)}):")
        for joint_info in joints_info:
            writer.print(f"  - {joint_info['name']}")
    
    # List all body names
    if bodies_info:
        writer.print(f"\nBody Names ({len(bodies_info)}):")
        for body_info in bodies_info:
            writer.print(f"  - {body_info['name']}")
    
    # List all actuator names
    if actuators_info:
        writer.print(f"\nActuator Names ({len(actuators_info)}):")
        for actuator_info in actuators_info:
            writer.print(f"  - {actuator_info['name']} (joint: {actuator_info['joint']})")
    
    writer.close()


def main():
    if not DIGIT_XML_V3.exists():
        print(f"❌ XML file not found: {DIGIT_XML_V3}")
        print(f"   Please ensure the file exists at: {DIGIT_XML_V3.absolute()}")
        return
    
    # Output file path
    output_file = "scripts/tools/digit_v3_parameters.txt"
    inspect_mjcf_file(DIGIT_XML_V3, output_file)


if __name__ == "__main__":
    main()

