"""
Main Application - Wheelchair Accessibility Navigation System
Demonstrates complete workflow from map creation to navigation
"""

import os
import sys
from map_creator import MapCreator, MapPixel, FuncID
from nav_utils import (
    NavigationGraph,
)

def create_sample_maps(maps_dir='maps'):
    """Create sample maps for demonstration"""
    print("Creating sample maps...")
    
    # Create maps directory if it doesn't exist
    os.makedirs(maps_dir, exist_ok=True)
    
    # 1. Campus map (outdoor area)
    print("\n1. Creating campus map...")
    campus = MapCreator(200, 200, map_id="campus")
    
    # Draw walkable outdoor paths
    for x in range(50, 150):
        for y in range(90, 110):
            campus.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=0,
                floor=0,
                func_id=FuncID.WALKABLE
            )
    
    # Add door to CS building at (80, 100)
    for x in range(78, 82):
        for y in range(98, 102):
            campus.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=0,
                floor=0,
                func_id=FuncID.DOOR,
                identifier=["cs_building_ground"]
            )
    
    # Add door to Library at (120, 100)
    for x in range(118, 122):
        for y in range(98, 102):
            campus.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=0,
                floor=0,
                func_id=FuncID.DOOR,
                identifier=["library_ground"]
            )
    
    campus.save_map(f'{maps_dir}/campus.bin')
    print("   ✓ Campus map created")
    
    # 2. CS Building - Ground Floor
    print("\n2. Creating CS building ground floor...")
    cs_ground = MapCreator(150, 150, map_id="cs_building_ground")
    
    # Walkable areas
    for x in range(20, 130):
        for y in range(20, 130):
            cs_ground.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=1,
                floor=0,
                func_id=FuncID.WALKABLE
            )
    
    # Entrance door back to campus
    for x in range(73, 77):
        for y in range(18, 22):
            cs_ground.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=1,
                floor=0,
                func_id=FuncID.DOOR,
                identifier=["campus"]
            )
    
    # Elevator to upper floors
    for x in range(73, 78):
        for y in range(73, 78):
            cs_ground.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=1,
                floor=0,
                func_id=FuncID.ELEVATOR,
                identifier=["cs_building_floor1", "cs_building_floor2"]
            )
    
    cs_ground.save_map(f'{maps_dir}/cs_building_ground.bin')
    print("   ✓ CS building ground floor created")
    
    # 3. CS Building - Floor 1
    print("\n3. Creating CS building floor 1...")
    cs_floor1 = MapCreator(150, 150, map_id="cs_building_floor1")
    
    # Walkable areas
    for x in range(20, 130):
        for y in range(20, 130):
            cs_floor1.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=1,
                floor=1,
                func_id=FuncID.WALKABLE
            )
    
    # Elevator
    for x in range(73, 78):
        for y in range(73, 78):
            cs_floor1.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=1,
                floor=1,
                func_id=FuncID.ELEVATOR,
                identifier=["cs_building_ground", "cs_building_floor2"]
            )
    
    cs_floor1.save_map(f'{maps_dir}/cs_building_floor1.bin')
    print("   ✓ CS building floor 1 created")
    
    # 4. CS Building - Floor 2
    print("\n4. Creating CS building floor 2...")
    cs_floor2 = MapCreator(150, 150, map_id="cs_building_floor2")
    
    # Walkable areas
    for x in range(20, 130):
        for y in range(20, 130):
            cs_floor2.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=1,
                floor=2,
                func_id=FuncID.WALKABLE
            )
    
    # Elevator
    for x in range(73, 78):
        for y in range(73, 78):
            cs_floor2.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=1,
                floor=2,
                func_id=FuncID.ELEVATOR,
                identifier=["cs_building_ground", "cs_building_floor1"]
            )
    
    cs_floor2.save_map(f'{maps_dir}/cs_building_floor2.bin')
    print("   ✓ CS building floor 2 created")
    
    # 5. Library - Ground Floor
    print("\n5. Creating library ground floor...")
    library = MapCreator(120, 120, map_id="library_ground")
    
    # Walkable areas
    for x in range(15, 105):
        for y in range(15, 105):
            library.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=2,
                floor=0,
                func_id=FuncID.WALKABLE
            )
    
    # Entrance door back to campus
    for x in range(58, 62):
        for y in range(13, 17):
            library.map_data[y][x] = MapPixel(
                cost=1,
                dept_id=2,
                floor=0,
                func_id=FuncID.DOOR,
                identifier=["campus"]
            )
    
    library.save_map(f'{maps_dir}/library_ground.bin')
    print("   ✓ Library ground floor created")
    
    print("\n✓ All sample maps created successfully!")
    return maps_dir


def test_navigation(maps_dir='maps'):
    """Test navigation functionality"""
    print("\n" + "="*60)
    print("TESTING NAVIGATION SYSTEM")
    print("="*60)
    
    # Load navigation graph
    nav_graph = NavigationGraph()
    nav_graph.load_maps(maps_dir)
    
    # Print graph information
    nav_graph.print_graph_info()
    
    # Validate connections
    print("\n" + "="*60)
    print("VALIDATING MAP CONNECTIONS")
    print("="*60)
    
    issues = validate_map_connections(nav_graph)
    
    if any(issues.values()):
        print("\nIssues found:")
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"\n{issue_type}:")
                for issue in issue_list:
                    print(f"  - {issue}")
    else:
        print("\n✓ All connections are valid!")
    
    # Test pathfinding
    print("\n" + "="*60)
    print("TEST 1: Campus to CS Building Floor 2")
    print("="*60)
    
    result = navigate_multi_map(
        nav_graph,
        start_map='campus',
        start_pos=(100, 100),
        end_map='cs_building_floor2',
        end_pos=(50, 50)
    )
    
    if result['success']:
        print("\n✓ Navigation successful!")
        for instruction in result['instructions']:
            print(instruction)
    else:
        print("\n✗ Navigation failed!")
        for instruction in result['instructions']:
            print(instruction)
    
    # Test 2: Within same building
    print("\n" + "="*60)
    print("TEST 2: CS Ground Floor to CS Floor 1")
    print("="*60)
    
    result2 = navigate_multi_map(
        nav_graph,
        start_map='cs_building_ground',
        start_pos=(30, 30),
        end_map='cs_building_floor1',
        end_pos=(100, 100)
    )
    
    if result2['success']:
        print("\n✓ Navigation successful!")
        for instruction in result2['instructions']:
            print(instruction)
    else:
        print("\n✗ Navigation failed!")
        for instruction in result2['instructions']:
            print(instruction)
    
    # Test 3: Check accessible destinations
    print("\n" + "="*60)
    print("TEST 3: Accessible Destinations from Campus")
    print("="*60)
    
    accessible = get_accessible_destinations(nav_graph, 'campus')
    print(f"\nFrom 'campus', you can reach:")
    for dest in accessible:
        print(f"  - {dest}")


def interactive_mode():
    """Interactive mode for creating and testing maps"""
    print("\n" + "="*60)
    print("WHEELCHAIR ACCESSIBILITY MAP SYSTEM")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Create sample maps")
        print("2. Create new map")
        print("3. Edit existing map")
        print("4. Test navigation")
        print("5. View graph info")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            maps_dir = input("Maps directory [maps]: ").strip() or 'maps'
            create_sample_maps(maps_dir)
        
        elif choice == '2':
            map_id = input("Map ID: ").strip()
            if not map_id:
                print("Error: Map ID required")
                continue
            
            width = int(input("Width [200]: ").strip() or "200")
            height = int(input("Height [200]: ").strip() or "200")
            
            creator = MapCreator(width, height, map_id=map_id)
            creator.run()
            
            # Save prompt
            save = input("Save map? (y/n): ").strip().lower()
            if save == 'y':
                maps_dir = input("Maps directory [maps]: ").strip() or 'maps'
                os.makedirs(maps_dir, exist_ok=True)
                filename = f"{maps_dir}/{map_id}.bin"
                creator.save_map(filename)
                print(f"✓ Saved to {filename}")
        
        elif choice == '3':
            maps_dir = input("Maps directory [maps]: ").strip() or 'maps'
            map_id = input("Map ID to edit: ").strip()
            
            filename = f"{maps_dir}/{map_id}.bin"
            if not os.path.exists(filename):
                print(f"Error: {filename} not found")
                continue
            
            creator = MapCreator(map_id=map_id)
            creator.load_map(filename)
            creator.run()
            
            # Save prompt
            save = input("Save changes? (y/n): ").strip().lower()
            if save == 'y':
                creator.save_map(filename)
                print(f"✓ Saved to {filename}")
        
        elif choice == '4':
            maps_dir = input("Maps directory [maps]: ").strip() or 'maps'
            test_navigation(maps_dir)
        
        elif choice == '5':
            maps_dir = input("Maps directory [maps]: ").strip() or 'maps'
            nav_graph = NavigationGraph()
            nav_graph.load_maps(maps_dir)
            nav_graph.print_graph_info()
        
        elif choice == '6':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid option")


def main():
    """Main entry point"""
    print("="*60)
    print("WHEELCHAIR ACCESSIBILITY NAVIGATION SYSTEM")
    print("="*60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'create-samples':
            maps_dir = sys.argv[2] if len(sys.argv) > 2 else 'maps'
            create_sample_maps(maps_dir)
        
        elif command == 'test':
            maps_dir = sys.argv[2] if len(sys.argv) > 2 else 'maps'
            test_navigation(maps_dir)
        
        elif command == 'create':
            if len(sys.argv) < 3:
                print("Usage: python main.py create <map_id> [width] [height]")
                return
            
            map_id = sys.argv[2]
            width = int(sys.argv[3]) if len(sys.argv) > 3 else 200
            height = int(sys.argv[4]) if len(sys.argv) > 4 else 200
            
            creator = MapCreator(width, height, map_id=map_id)
            creator.run()
        
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python main.py create-samples [maps_dir]")
            print("  python main.py test [maps_dir]")
            print("  python main.py create <map_id> [width] [height]")
            print("  python main.py  (interactive mode)")
    
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
