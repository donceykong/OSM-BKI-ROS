"""Launch MCD node and OSM visualizer node together as independent nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
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
    
    osm_dataset_arg = DeclareLaunchArgument(
        'osm_dataset',
        default_value='osm_visualizer',
        description='OSM visualizer config dataset name'
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
        description='OSM prior decay distance in meters'
    )
    
    return LaunchDescription([
        pkg_arg,
        method_arg,
        dataset_arg,
        osm_dataset_arg,
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
    osm_dataset = context.launch_configurations.get('osm_dataset', 'osm_visualizer')
    color_mode = context.launch_configurations.get('color_mode', 'semantic')
    osm_file = context.launch_configurations.get('osm_file', '')
    osm_origin_lat = context.launch_configurations.get('osm_origin_lat', '0.0')
    osm_origin_lon = context.launch_configurations.get('osm_origin_lon', '0.0')
    osm_decay_meters = context.launch_configurations.get('osm_decay_meters', '2.0')
    
    # Resolve the source package directory so data/config are read from src/,
    # not from the installed share directory.
    # Primary: navigate from install share dir up to workspace root.
    # Fallback: use this file's location (works when running directly from src/).
    pkg_share_dir = get_package_share_directory('semantic_bki')
    ws_root = os.path.abspath(os.path.join(pkg_share_dir, '..', '..', '..', '..'))
    pkg_src_dir = os.path.join(ws_root, 'src', 'BKISemanticMapping')
    if not os.path.isdir(os.path.join(pkg_src_dir, 'config')):
        pkg_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    method_config_path = os.path.join(pkg_src_dir, 'config', 'methods', f'{method}.yaml')
    data_config_path = os.path.join(pkg_src_dir, 'config', 'datasets', f'{dataset}.yaml')
    osm_config_path = os.path.join(pkg_src_dir, 'config', 'datasets', f'{osm_dataset}.yaml')
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
    
    # MCD node (scan processing) - now with OSM prior support
    # color_mode, osm_decay_meters, osm_file, osm_origin_* come from dataset config (mcd.yaml) by default
    mcd_params = [
        {'dir': data_dir_path},
        {'calibration_file': calib_file_path},
        method_config_path,
        data_config_path
    ]
    # Allow launch args to override OSM settings when explicitly provided (e.g. osm_file:=other.osm)
    if osm_file or osm_origin_lat != '0.0' or osm_origin_lon != '0.0':
        mcd_params.append({
            'osm_file': osm_file,
            'osm_origin_lat': float(osm_origin_lat),
            'osm_origin_lon': float(osm_origin_lon),
        })
    
    mcd_node = Node(
        package='semantic_bki',
        executable='mcd_node',
        name='mcd_node',
        output='screen',
        parameters=mcd_params
    )
    
    # OSM visualizer node (independent, publishes OSM buildings/roads/etc. as markers)
    # Pass data_dir so pose file can be found relative to it
    osm_node = Node(
        package='semantic_bki',
        executable='osm_visualizer_node',
        name='osm_visualizer_node',
        output='screen',
        parameters=[
            osm_config_path,
            {'data_dir': data_dir_path}
        ]
    )
    
    return [rviz_node, mcd_node, osm_node]
