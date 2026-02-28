from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    pkg_arg = DeclareLaunchArgument(
        'pkg',
        default_value='semantic_bki',
        description='Package name'
    )
    
    method_arg = DeclareLaunchArgument(
        'method',
        default_value='semantic_bki',
        description='Method name'
    )
    
    dataset_arg = DeclareLaunchArgument(
        'dataset',
        default_value='mcd',
        description='Dataset name'
    )
    
    color_mode_arg = DeclareLaunchArgument(
        'color_mode',
        default_value='semantic',
        description='Visualization color mode: semantic, osm_building, osm_road, osm_grassland, osm_tree, osm_parking, osm_fence, osm_blend (all priors blended)'
    )
    
    osm_file_arg = DeclareLaunchArgument(
        'osm_file',
        default_value='',
        description='Path to OSM file (relative to data dir or absolute) for voxel priors. Empty to disable.'
    )
    
    osm_origin_lat_arg = DeclareLaunchArgument(
        'osm_origin_lat',
        default_value='0.0',
        description='OSM origin latitude (degrees)'
    )
    
    osm_origin_lon_arg = DeclareLaunchArgument(
        'osm_origin_lon',
        default_value='0.0',
        description='OSM origin longitude (degrees)'
    )
    
    osm_decay_meters_arg = DeclareLaunchArgument(
        'osm_decay_meters',
        default_value='2.0',
        description='OSM prior decay distance in meters (distance over which prior drops from 1 to 0)'
    )
    
    return LaunchDescription([
        pkg_arg,
        method_arg,
        dataset_arg,
        color_mode_arg,
        osm_file_arg,
        osm_origin_lat_arg,
        osm_origin_lon_arg,
        osm_decay_meters_arg,
        OpaqueFunction(function=launch_setup)
    ])


def launch_setup(context):
    # Get launch argument values
    method = context.launch_configurations.get('method', 'semantic_bki')
    dataset = context.launch_configurations.get('dataset', 'mcd')
    color_mode = context.launch_configurations.get('color_mode', 'semantic')
    osm_file = context.launch_configurations.get('osm_file', '')
    osm_origin_lat = context.launch_configurations.get('osm_origin_lat', '0.0')
    osm_origin_lon = context.launch_configurations.get('osm_origin_lon', '0.0')
    osm_decay_meters = context.launch_configurations.get('osm_decay_meters', '2.0')
    
    # Resolve source directory from the install path so data is read from src/
    # install/<pkg>/share/<pkg> -> workspace root -> src/BKISemanticMapping
    pkg_share_dir = get_package_share_directory('semantic_bki')
    ws_root = os.path.abspath(os.path.join(pkg_share_dir, '..', '..', '..', '..'))
    pkg_src_dir = os.path.join(ws_root, 'src', 'BKISemanticMapping')
    
    method_config_path = os.path.join(pkg_src_dir, 'config', 'methods', f'{method}.yaml')
    data_config_path = os.path.join(pkg_src_dir, 'config', 'datasets', f'{dataset}.yaml')
    data_dir_path = os.path.join(pkg_src_dir, 'data', dataset)
    calib_file_path = os.path.join(pkg_src_dir, 'data', dataset, 'hhs_calib.yaml')
    rviz_config_path = os.path.join(pkg_src_dir, 'rviz', 'mcd_node.rviz')
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    
    # MCD node parameters
    mcd_params = [
        {'dir': data_dir_path},
        {'calibration_file': calib_file_path},  # Pass as string parameter, not a parameter file
        {'osm_file': osm_file},  # OSM file path
        {'osm_origin_lat': float(osm_origin_lat)},  # OSM origin latitude
        {'osm_origin_lon': float(osm_origin_lon)},  # OSM origin longitude
        method_config_path,
        data_config_path
    ]
    
    # MCD node
    mcd_node = Node(
        package='semantic_bki',
        executable='mcd_node',
        name='mcd_node',
        output='screen',
        parameters=mcd_params
    )
    
    return [rviz_node, mcd_node]
